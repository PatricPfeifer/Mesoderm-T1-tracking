"""Manual cell correction (draw / delete / undo) for a Labels layer.

Ported from napariCellFlow (Artur Ruppel): ``CellCorrectionWidget`` verbatim, with a
slimmed ``DataManager`` and ``VisualizationManager`` that keep only what the widget uses
(the stack bookkeeping and the "Segmentation" Labels layer). napariCellFlow is no longer
needed. Licence of the original: see its LICENCE.md / pyproject.toml.
"""
import logging
from threading import Lock
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QApplication,
    QShortcut, QSizePolicy
)

logger = logging.getLogger("napari_tissuekit.correction")


class DataManager:
    """Holds the segmentation stack the correction widget edits (frame-wise or whole)."""

    def __init__(self):
        self._updating = False
        self._lock = Lock()
        self._segmentation_data = None
        self._num_frames = None
        self._frame_states = {}
        self._initialized = False

    def initialize_stack(self, num_frames: int) -> None:
        with self._lock:
            self._num_frames = num_frames
            self._frame_states = {i: {"modified": False, "last_update": None} for i in range(num_frames)}
            self._initialized = True

    @property
    def segmentation_data(self):
        with self._lock:
            return self._segmentation_data

    @segmentation_data.setter
    def segmentation_data(self, value):
        """Set the whole stack, or one frame given as ``(frame_array, frame_index)``."""
        if self._updating:
            return
        with self._lock:
            try:
                self._updating = True
                if isinstance(value, tuple):
                    self._update_single_frame(*value)
                else:
                    self._update_full_stack(value)
            finally:
                self._updating = False

    def _update_single_frame(self, frame_data, frame_index):
        if frame_index >= self._num_frames:
            raise ValueError(f"Frame index {frame_index} out of bounds")
        if self._segmentation_data is None:
            self._segmentation_data = np.zeros((self._num_frames, *frame_data.shape), dtype=frame_data.dtype)
        elif frame_data.shape != self._segmentation_data.shape[1:]:
            raise ValueError(
                f"Frame shape mismatch: expected {self._segmentation_data.shape[1:]}, got {frame_data.shape}"
            )
        self._segmentation_data[frame_index] = frame_data.copy()
        self._frame_states[frame_index] = {"modified": True, "last_update": np.datetime64("now")}

    def _update_full_stack(self, stack_data):
        if stack_data is None:
            self._segmentation_data = None
            self._frame_states.clear()
            return
        if stack_data.ndim == 2:
            stack_data = stack_data[np.newaxis, ...]
        if stack_data.ndim != 3:
            raise ValueError(f"Invalid data dimensions: {stack_data.shape}")
        if stack_data.shape[0] < self._num_frames:
            padded = np.zeros((self._num_frames, *stack_data.shape[1:]), dtype=stack_data.dtype)
            padded[: stack_data.shape[0]] = stack_data
            self._segmentation_data = padded
        else:
            self._segmentation_data = stack_data.copy()
        self._frame_states = {
            i: {"modified": False, "last_update": np.datetime64("now")}
            for i in range(self._segmentation_data.shape[0])
        }


class VisualizationManager:
    """Owns the "Segmentation" Labels layer and writes edits back to it."""

    def __init__(self, viewer, data_manager: DataManager):
        self.viewer = viewer
        self.data_manager = data_manager
        self.tracking_layer = None
        self._updating = False
        self._layer_lock = Lock()
        try:
            self.viewer.layers.events.removed.connect(self._handle_layer_removal)
        except Exception as exc:
            logger.warning("Could not watch layer removal: %s", exc)

    def update_tracking_visualization(self, data) -> None:
        """Show ``data`` (whole stack, or ``(frame_array, frame_index)``) on the layer."""
        if self._updating:
            return
        try:
            self._updating = True
            with self._layer_lock:
                update_data = self._prepare_update_data(data)
                with self.viewer.events.blocker_all():
                    if self.tracking_layer is not None and self.tracking_layer in self.viewer.layers:
                        self.tracking_layer.data = update_data
                        self.tracking_layer.refresh()
                    else:
                        self.tracking_layer = self.viewer.add_labels(
                            update_data, name="Segmentation", opacity=0.5, visible=True
                        )
        except Exception as exc:
            logger.error("Failed to update visualization: %s", exc, exc_info=True)
        finally:
            self._updating = False

    def _prepare_update_data(self, data):
        num_frames = self.data_manager._num_frames
        if isinstance(data, tuple):
            frame_data, frame_index = data
            if not isinstance(frame_data, np.ndarray):
                raise ValueError("Frame data must be a numpy array")
            if frame_index >= num_frames:
                raise ValueError(f"Frame index {frame_index} exceeds stack size {num_frames}")
            if self.tracking_layer is None:
                out = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
            else:
                current = self.tracking_layer.data
                out = current.copy()
                if current.shape[0] < num_frames:
                    out = np.zeros((num_frames, *frame_data.shape), dtype=frame_data.dtype)
                    out[: current.shape[0]] = current
            out[frame_index] = frame_data
            return out
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.shape[0] < num_frames:
            padded = np.zeros((num_frames, *data.shape[1:]), dtype=data.dtype)
            padded[: data.shape[0]] = data
            return padded
        return data.copy()

    def _handle_layer_removal(self, event):
        if event.value == self.tracking_layer:
            with self._layer_lock:
                if event.value not in self.viewer.layers:
                    self.tracking_layer = None



class ActionType(Enum):
    DRAW = "draw"
    DELETE = "delete"


@dataclass
class UndoAction:
    """Represents a single undoable action with full stack context"""
    action_type: ActionType
    frame: Optional[int]  # Frame number for 3D stacks, None for 2D
    previous_state: np.ndarray  # Previous state of the entire stack
    description: str  # Description of the action
    affected_cell_ids: set  # Set of cell IDs affected by this action


class CellCorrectionWidget(QWidget):
    """Widget for interactive cell correction in napari."""

    correction_made = Signal(np.ndarray)  # Emitted when masks are modified
    MAX_UNDO_STEPS = 20
    MIN_DRAWING_POINTS = 10
    DRAG_START_THRESHOLD = 4  # px — left-click delete vs left-drag draw
    _EDIT_KEYS = frozenset({Qt.Key_Control, Qt.Key_Alt})

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager", parent: QWidget = None):
        super().__init__(parent)
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        # Add flags for state management
        self._updating = False
        self._loading_external = False
        self._undo_in_progress = False

        # Initialize masks layer
        self.masks_layer = None
        if self.vis_manager.tracking_layer is not None:
            self.masks_layer = self.vis_manager.tracking_layer
            # Initialize next_cell_id based on existing data
            if self.masks_layer.data is not None:
                self.next_cell_id = int(np.max(self.masks_layer.data)) + 1
            else:
                self.next_cell_id = 1
        elif hasattr(self.data_manager, 'segmentation_data') and self.data_manager.segmentation_data is not None:
            self.set_masks_layer(self.data_manager.segmentation_data)
            # Initialize next_cell_id based on existing data
            self.next_cell_id = int(np.max(self.data_manager.segmentation_data)) + 1
        else:
            self.next_cell_id = 1

        # Store full stack reference
        self._full_stack = None
        if self.masks_layer is not None:
            self._update_full_stack(self.masks_layer.data)

        # Connect to viewer layer events
        self.viewer.layers.events.inserted.connect(self._handle_layer_added)
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

        self.drawing_layer = None
        self.current_frame = 0
        self.is_drawing = False
        self.drawing_points = []
        self.next_cell_id = 1
        self.selected_cell = None
        self.drawing_started = False
        self.start_point = None
        self.edit_modifier_pressed = False
        self._edit_keys_down = set()
        self.toggle_state = False

        # Enhanced undo history
        self.undo_stack = deque(maxlen=self.MAX_UNDO_STEPS)

        # Register undo with napari (Control-Z and Meta-Z / Cmd+Z on Mac)
        self.viewer.bind_key('Control-Z', self.undo_last_action)
        self.viewer.bind_key('Meta-Z', self.undo_last_action)

        self._setup_ui()
        self._connect_events()
        self._setup_shortcuts()
        QApplication.instance().installEventFilter(self)

    def _setup_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.status_label)

        # Updated instructions
        instructions = QLabel(
            "Cell Editing Controls:\n"
            "- Hold Option (or Ctrl) to enable cell editing\n"
            "- Left click to delete cells\n"
            "- Left click and drag to draw cell boundary\n"
            "- Return to start point (marked with circle) to complete the cell\n"
            "- Cmd+Z / Ctrl+Z to undo last action"
        )
        instructions.setWordWrap(True)
        instructions.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(instructions)

    def _on_mouse_move(self, viewer, event):
        """Optimized mouse movement handler with faster closing detection."""
        if not self.is_drawing or not self.drawing_started:
            return

        coords = np.round(viewer.cursor.position).astype(int)[-2:]
        self._continue_drawing(coords)

    def _continue_drawing(self, coords):
        """Append drawing points and close the polygon when near the start."""
        # Early exit if not enough points for closing
        if len(self.drawing_points) < self.MIN_DRAWING_POINTS:
            self.drawing_points.append(coords)
            if len(self.drawing_points) % 5 == 0:
                self._update_drawing_preview()
            return

        # Fast distance check to start point using numpy
        distance_to_start = np.linalg.norm(coords - self.start_point)

        # Calculate closure threshold based on image width (same as in preview)
        image_width = (self.masks_layer.data.shape[2]
                       if len(self.masks_layer.data.shape) == 3
                       else self.masks_layer.data.shape[1])
        closure_radius = max(3, int(3 * image_width / 200))

        # Check for closure
        if distance_to_start < closure_radius:
            # Quick validation of shape before closing
            if len(self.drawing_points) >= self.MIN_DRAWING_POINTS:
                # Add final point to close the shape cleanly
                self.drawing_points.append(self.start_point)
                self._finish_drawing()
                return

        # Point throttling for performance
        if self.drawing_points:
            last_point = self.drawing_points[-1]
            min_distance = 2  # Minimum pixels between points
            if np.linalg.norm(coords - last_point) >= min_distance:
                self.drawing_points.append(coords)
                if len(self.drawing_points) % 5 == 0:
                    self._update_drawing_preview()

    def _finish_drawing(self):
        """Optimized drawing completion."""
        if not self.drawing_points or len(self.drawing_points) < 3:
            logger.debug("CellCorrection: Not enough points to complete drawing")
            self._clear_drawing()
            return

        if self._updating:
            return

        try:
            self._updating = True
            current_frame = int(self.viewer.dims.point[0])

            if not self.vis_manager.tracking_layer:
                raise ValueError("No tracking layer available")

            # Get current state efficiently
            prev_state = self.vis_manager.tracking_layer.data.copy()
            if prev_state.ndim == 2:
                prev_state = prev_state[np.newaxis, ...]

            # Fast max calculation for next cell ID
            current_max = int(np.max(prev_state)) if prev_state.size > 0 else 0
            self.next_cell_id = max(self.next_cell_id, current_max + 1)

            # Create undo action
            action = UndoAction(
                action_type=ActionType.DRAW,
                frame=current_frame,
                previous_state=prev_state,
                description=f"Draw new cell {self.next_cell_id}",
                affected_cell_ids={self.next_cell_id}
            )
            self.undo_stack.append(action)

            # Efficient shape getting
            frame_shape = (prev_state.shape[1:] if prev_state.ndim == 3
                           else prev_state.shape)

            # Create cell mask with optimized array operations
            new_cell_mask = np.zeros(frame_shape, dtype=np.uint8)
            points = np.array(self.drawing_points)
            points = np.clip(points, 0, np.array(frame_shape) - 1)
            cv2.fillPoly(new_cell_mask, [points[:, ::-1].astype(np.int32)], 1)

            # Get current frame data efficiently
            if prev_state.ndim == 3:
                current_frame_data = prev_state[current_frame].copy()
            else:
                current_frame_data = prev_state.copy()

            # Optimized mask updates using boolean operations
            empty_mask = (current_frame_data == 0)
            add_mask = np.logical_and(new_cell_mask > 0, empty_mask)
            current_frame_data[add_mask] = self.next_cell_id

            # Update data and visualization
            if prev_state.ndim == 3:
                self.data_manager.segmentation_data = (current_frame_data, current_frame)
            else:
                self.data_manager.segmentation_data = current_frame_data

            self.vis_manager.update_tracking_visualization(
                (current_frame_data, current_frame) if prev_state.ndim == 3
                else current_frame_data
            )

            self.status_label.setText(f"Added new cell {self.next_cell_id}")
            self.next_cell_id += 1

        except Exception as e:
            logger.error(f"CellCorrection: Error finishing drawing: {e}")
            raise
        finally:
            self._updating = False
            self.drawing_started = False
            self.start_point = None
            self._clear_drawing()

    def _begin_drawing_at(self, coords):
        """Start a new cell-boundary drawing at coords (preview + start circle)."""
        if self.drawing_layer is None:
            shape = self.masks_layer.data.shape
            base_shape = shape[1:] if len(shape) == 3 else shape
            preview = np.zeros(base_shape, dtype=np.uint8)
            self.drawing_layer = self.viewer.add_labels(
                preview,
                name='Drawing Preview',
                opacity=0.8,
                visible=True
            )

        preview = np.zeros_like(self.drawing_layer.data)
        image_width = preview.shape[1]
        circle_radius = max(3, int(3 * image_width / 200))
        cv2.circle(preview, (coords[1], coords[0]), circle_radius, 2, -1)
        self.drawing_layer.data = preview

        self.drawing_started = True
        self.start_point = coords
        self.drawing_points = [coords]

    def _on_mouse_drag(self, viewer, event):
        """Left-click deletes; left-drag draws. Napari drag generator lifecycle."""
        try:
            if self.masks_layer is None and self.vis_manager.tracking_layer is not None:
                self.masks_layer = self.vis_manager.tracking_layer

            if self.masks_layer is None:
                return

            if event.button != Qt.LeftButton or not self.is_drawing:
                return

            press_coords = np.round(viewer.cursor.position).astype(int)[-2:]

            # Continue an in-progress polygon during this drag
            if self.drawing_started:
                self._continue_drawing(press_coords)
                yield
                while event.type == 'mouse_move':
                    coords = np.round(viewer.cursor.position).astype(int)[-2:]
                    self._continue_drawing(coords)
                    yield
                return

            # Provisional press: wait to distinguish click (delete) vs drag (draw)
            yield
            started_draw = False
            while event.type == 'mouse_move':
                coords = np.round(viewer.cursor.position).astype(int)[-2:]
                if not started_draw:
                    dist = np.linalg.norm(
                        coords.astype(float) - press_coords.astype(float)
                    )
                    if dist >= self.DRAG_START_THRESHOLD:
                        started_draw = True
                        self._begin_drawing_at(press_coords)
                        self._continue_drawing(coords)
                else:
                    self._continue_drawing(coords)
                yield

            # Release with little/no movement → delete cell under press
            if not started_draw and not self.drawing_started:
                self._delete_cell_at_position(press_coords)

        except Exception as e:
            logger.error(f"CellCorrection: Error in mouse drag: {e}", exc_info=True)
    def _update_drawing_preview(self):
        """Fast drawing preview update with batched operations and dynamic scaling."""
        if not self.drawing_points or len(self.drawing_points) < 2:
            return

        try:
            # Create preview array only when needed
            if self.drawing_layer is None:
                shape = self.masks_layer.data.shape
                base_shape = shape[1:] if len(shape) == 3 else shape
                preview = np.zeros(base_shape, dtype=np.uint8)
                self.drawing_layer = self.viewer.add_labels(
                    preview,
                    name='Drawing Preview',
                    opacity=0.8,
                    visible=True
                )
            else:
                preview = np.zeros_like(self.drawing_layer.data)

            # Calculate scaling based on image width
            image_width = preview.shape[1]
            thickness = max(1, int(image_width / 200))  # Keep dynamic thickness scaling
            circle_radius = max(3, int(3 * image_width / 200))  # Keep dynamic circle radius

            # Batch process points for efficiency
            points = np.array(self.drawing_points)[:, ::-1]  # Single array conversion

            # Draw only last N points for efficiency on very long paths
            if len(points) > 200:
                points = points[-200:]

            # Draw polyline with scaled thickness
            cv2.polylines(preview, [points.astype(np.int32)], False, 1, thickness)

            # Draw start point circle with scaled radius
            if self.start_point is not None:
                cv2.circle(preview,
                           (self.start_point[1], self.start_point[0]),
                           circle_radius, 2, -1)

            self.drawing_layer.data = preview

        except Exception as e:
            logger.error(f"Error in drawing preview: {e}")

    def _clear_drawing(self):
        """Fast drawing cleanup."""
        if self.drawing_layer is not None:
            self.viewer.layers.remove(self.drawing_layer)
            self.drawing_layer = None

        self.drawing_points = []
        self.drawing_started = False
        self.start_point = None

    def _update_ui_state(self):
        """Update UI elements based on current state"""
        try:
            # Simple status update
            if self.is_drawing:
                self.status_label.setText("Cell editing enabled")
            else:
                self.status_label.setText(
                    "Hold Option (or Ctrl) to enable cell editing"
                )

        except Exception as e:
            logger.error(f"CellCorrection: Error updating UI state: {e}")

    def _connect_events(self):
        """Connect all event handlers"""
        # Remove any existing callbacks first
        if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        if self._on_mouse_move in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        if self._on_mouse_wheel in self.viewer.mouse_wheel_callbacks:
            self.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)

        # Add mouse callbacks explicitly to viewer
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_drag)
        self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self.viewer.mouse_wheel_callbacks.append(self._on_mouse_wheel)

    def _update_drawing_state(self):
        """Update drawing state with proper synchronization."""
        try:
            new_drawing_state = self.edit_modifier_pressed

            # Only update if state actually changes
            if new_drawing_state == self.is_drawing:
                return

            self.is_drawing = new_drawing_state

            # Only update UI if we're not actively drawing
            if not self.drawing_started:
                with self.viewer.events.blocker_all():
                    self._update_ui_state()
                    if not new_drawing_state:
                        self._clear_drawing()

            # Single refresh at the end if needed
            if self.masks_layer and self.masks_layer in self.viewer.layers:
                self.masks_layer.refresh()

        except Exception as e:
            logger.error(f"CellCorrection: Error updating drawing state: {e}")

    def _delete_cell_at_position(self, coords):
        """Delete cell at the given coordinates."""
        if self._updating or not self._validate_coords(coords):
            return

        current_mask = self._get_current_frame_mask()
        if current_mask is None:
            return

        cell_id = current_mask[coords[0], coords[1]]
        if cell_id > 0:
            try:
                # Store current state for undo
                if self.vis_manager.tracking_layer is not None:
                    prev_state = self.vis_manager.tracking_layer.data.copy()
                    if prev_state.ndim == 2:
                        prev_state = prev_state[np.newaxis, ...]

                    action = UndoAction(
                        action_type=ActionType.DELETE,
                        frame=int(self.viewer.dims.point[0]),
                        previous_state=prev_state,
                        description=f"Delete cell {cell_id}",
                        affected_cell_ids={cell_id}
                    )
                    self.undo_stack.append(action)

                self._updating = True

                # Create a copy of just the current frame
                new_frame = current_mask.copy()
                new_frame[new_frame == cell_id] = 0

                # Get current frame index
                current_slice = int(self.viewer.dims.point[0])

                # Update data manager
                self.data_manager.segmentation_data = (new_frame, current_slice)

                # Update visualization
                self.vis_manager.update_tracking_visualization(
                    (new_frame, current_slice)
                )

                self.status_label.setText(f"Deleted cell {cell_id}")

            finally:
                self._updating = False

    def eventFilter(self, watched_object, event):
        """Global event filter to catch key events regardless of focus"""
        if event.type() not in (event.KeyPress, event.KeyRelease):
            return False

        key = event.key()
        if key not in self._EDIT_KEYS:
            return False

        try:
            if event.type() == event.KeyPress:
                self._edit_keys_down.add(key)
            else:
                self._edit_keys_down.discard(key)
            self.edit_modifier_pressed = bool(self._edit_keys_down)
            self._update_drawing_state()
            return False

        except Exception as e:
            logger.error(f"CellCorrection: Error handling edit modifier: {e}")
            return False

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts"""
        try:
            # Create shortcut with the viewer as parent to ensure global scope
            self.undo_shortcut = QShortcut(QKeySequence.Undo, self.viewer.window())
            self.undo_shortcut.activated.connect(self.undo_last_action)
            logger.debug("Undo shortcut setup complete")
        except Exception as e:
            logger.error(f"Error setting up shortcuts: {e}")

    def _store_undo_state(self, description: str):
        """Store current state for undo"""
        if self._updating or self._undo_in_progress:
            return

        try:
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                return

            # Get current frame
            current_frame = int(self.viewer.dims.point[0]) if len(self.masks_layer.data.shape) == 3 else None

            # Store the entire stack state
            current_state = self.masks_layer.data.copy()
            if current_state.ndim == 2:
                current_state = current_state[np.newaxis, ...]

            # Determine action type based on description
            action_type = ActionType.DELETE if "Delete" in description else ActionType.DRAW

            # Get affected cell IDs
            affected_ids = set()
            if self.selected_cell is not None:
                affected_ids.add(self.selected_cell)
            elif "Draw new cell" in description:
                affected_ids.add(self.next_cell_id)

            # Create undo action
            action = UndoAction(
                action_type=action_type,
                frame=current_frame,
                previous_state=current_state,
                description=description,
                affected_cell_ids=affected_ids
            )

            self.undo_stack.append(action)
            logger.debug(f"Stored undo state: {description} ({action_type})")

        except Exception as e:
            logger.error(f"Error storing undo state: {e}")

    def undo_last_action(self, event=None):
        """Undo the last action with proper state restoration.

        Parameters
        ----------
        event : napari.utils.events.Event, optional
            The event triggering the undo action, by default None
        """
        if not self.undo_stack:
            return

        try:
            self._undo_in_progress = True
            self._updating = True

            action = self.undo_stack.pop()

            # Restore the state
            restored_state = action.previous_state.copy()

            # Update the layer
            if restored_state.ndim == 3 and restored_state.shape[0] == 1:
                restored_state = restored_state[0]  # Convert back to 2D if needed

            # Update through visualization manager to maintain consistency
            self.vis_manager.update_tracking_visualization(restored_state)

            # Update data manager
            self.data_manager.segmentation_data = restored_state

            # Emit correction signal
            self.correction_made.emit(restored_state)

            self.status_label.setText(f"Undid: {action.description}")
            logger.info(f"Undid action: {action.description} ({action.action_type})")


        except Exception as e:
            logger.error(f"Error during undo: {e}")
            raise
        finally:
            self._undo_in_progress = False
            self._updating = False
            self._clear_drawing()

    def _init_with_external_data(self, data: np.ndarray):
        """Initialize widget with externally loaded data"""
        if self._updating:
            return

        try:
            self._updating = True

            # Ensure proper dimensionality
            if data.ndim == 2:
                data = data[np.newaxis, ...]

            # Initialize data manager
            if not self.data_manager._initialized:
                self.data_manager.initialize_stack(data.shape[0])
            self.data_manager.segmentation_data = data

            # Update visualization
            self.vis_manager.update_tracking_visualization(data)

            # Update internal references
            self.masks_layer = self.vis_manager.tracking_layer
            self.next_cell_id = int(data.max()) + 1

            # Clear undo stack when loading new data
            self.undo_stack.clear()

        except Exception as e:
            logger.error(f"Failed to initialize with external data: {e}")
            raise
        finally:
            self._updating = False

    def _update_full_stack(self, data: np.ndarray):
        """Update the internal full stack reference"""
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        self._full_stack = data.copy()

    def _initialize_or_recover_layer(self):
        """Initialize or recover the masks layer."""
        try:
            if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                logger.debug("CellCorrection: Attempting to recover masks layer")
                if self.vis_manager.tracking_layer is not None:
                    self.masks_layer = self.vis_manager.tracking_layer
                    logger.debug("CellCorrection: Recovered masks layer from visualization manager")
                else:
                    logger.debug("CellCorrection: No tracking layer available")
        except Exception as e:
            logger.error(f"CellCorrection: Error initializing/recovering layer: {e}")

    def _create_cell_mask(self, frame_shape):
        """Create a mask for the new cell"""
        new_cell_mask = np.zeros(frame_shape, dtype=np.uint8)
        points = np.array(self.drawing_points + [self.start_point])
        points = np.clip(points, 0, np.array(frame_shape) - 1).astype(np.int32)
        cv2.fillPoly(new_cell_mask, [points[:, ::-1]], 1)
        return new_cell_mask

    def _handle_layer_added(self, event):
        """Handle when a new layer is added to the viewer"""
        if self._updating:
            logger.debug("Skipping layer addition - updating in progress")
            return

        layer = event.value
        logger.debug(f"New layer added: {layer.name}, type: {type(layer)}")

        if isinstance(layer, napari.layers.Labels):
            try:
                self._updating = True
                logger.debug("Processing new Labels layer")

                # Initialize with the new layer
                self.masks_layer = layer
                if layer.name != 'Drawing Preview':
                    layer.name = 'Segmentation'
                    logger.debug("Renamed layer to Segmentation")

                # Store full data
                data = layer.data
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                logger.debug(f"Layer data shape: {data.shape}")
                logger.debug(f"Unique values in data: {np.unique(data)}")

                # Update data manager
                if not self.data_manager._initialized:
                    logger.debug("Initializing data manager")
                    self.data_manager.initialize_stack(data.shape[0])
                self.data_manager.segmentation_data = data

                # Update visualization manager
                if layer.name != 'Drawing Preview':
                    logger.debug("Setting tracking layer in visualization manager")
                    self.vis_manager.tracking_layer = layer

                # Update next cell ID
                self.next_cell_id = int(data.max()) + 1
                logger.debug(f"Set next cell ID to: {self.next_cell_id}")

            except Exception as e:
                logger.error(f"Failed to initialize with external layer: {e}", exc_info=True)
            finally:
                self._updating = False
                logger.debug("Layer initialization complete")

    def _on_mouse_wheel(self, viewer, event):
        """Handle mouse wheel events with proper state synchronization."""
        if self._updating:
            return

        try:
            self._updating = True
            current_frame = int(self.viewer.dims.point[0])

            # Cancel any active drawing operation
            if self.drawing_started:
                self.drawing_started = False
                self.drawing_points = []
                self.start_point = None
                self.is_drawing = False
                self.edit_modifier_pressed = False
                self._edit_keys_down.clear()
                logger.debug("CellCorrection: Cancelled drawing due to frame change")

            # Clear any drawing preview
            self._clear_drawing()

            # Update masks layer if needed
            if self.masks_layer is not None and self.masks_layer in self.viewer.layers:
                # Ensure proper data shape
                if len(self.masks_layer.data.shape) == 3:
                    self.masks_layer.refresh()
                    # Update visualization manager
                    if self.vis_manager.tracking_layer is not None:
                        self.vis_manager.tracking_layer.refresh()

            # Update UI state
            self._update_ui_state()

        except Exception as e:
            logger.error(f"Error during mouse wheel: {e}")
        finally:
            self._updating = False

    def _validate_coords(self, coords):
        """Validate coordinates are within image bounds."""
        if self.masks_layer is None:
            return False

        try:
            # Get current frame shape
            if len(self.masks_layer.data.shape) == 3:
                shape = self.masks_layer.data.shape[1:]
            else:
                shape = self.masks_layer.data.shape

            # Validate coordinates
            return (0 <= coords[0] < shape[0] and
                    0 <= coords[1] < shape[1])

        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return False

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        if event.value == self.masks_layer:
            self.masks_layer = None
            self._full_masks = None

    def _handle_toggle_button(self, checked):
        """Handle toggle button state changes"""
        self.toggle_state = checked
        self._update_drawing_state()

    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self._updating:
            return

        try:
            self._updating = True
            logger.debug(f"Setting masks layer with shape {masks.shape}")

            # Ensure proper dimensionality
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]

            # Update through visualization manager first
            if self.vis_manager is not None:
                self.vis_manager.update_tracking_visualization(masks)
                self.masks_layer = self.vis_manager.tracking_layer
            else:
                # Create new layer if needed
                if self.masks_layer is None or self.masks_layer not in self.viewer.layers:
                    self.masks_layer = self.viewer.add_labels(
                        masks,
                        name='Segmentation',
                        opacity=0.5
                    )
                else:
                    self.masks_layer.data = masks

            # Update visualization manager reference
            if self.vis_manager is not None:
                self.vis_manager.tracking_layer = self.masks_layer

            # Set next cell ID
            self.next_cell_id = int(masks.max()) + 1

        except Exception as e:
            logger.error(f"Error setting masks layer: {str(e)}")
            raise
        finally:
            self._updating = False

    def _get_current_frame_mask(self):
        """Get mask for the current frame."""
        if self.masks_layer is None:
            return None

        if len(self.masks_layer.data.shape) == 3:
            # For 3D data, get current frame
            current_frame = int(self.viewer.dims.point[0])
            return self.masks_layer.data[current_frame]
        else:
            # For 2D data, return as is
            return self.masks_layer.data

    def cleanup(self):
        """Clean up resources and disconnect events"""
        try:
            # Remove application-wide event filter
            QApplication.instance().removeEventFilter(self)

            # Disconnect layer events
            self.viewer.layers.events.inserted.disconnect(self._handle_layer_added)
            self.viewer.layers.events.removed.disconnect(self._handle_layer_removal)

            # Remove mouse callbacks
            if hasattr(self.viewer, 'mouse_drag_callbacks'):
                if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
                    self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
            if hasattr(self.viewer, 'mouse_move_callbacks'):
                if self._on_mouse_move in self.viewer.mouse_move_callbacks:
                    self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
            if hasattr(self.viewer, 'mouse_wheel_callbacks'):
                if self._on_mouse_wheel in self.viewer.mouse_wheel_callbacks:
                    self.viewer.mouse_wheel_callbacks.remove(self._on_mouse_wheel)

            # Remove napari shortcut bindings
            self.viewer.bind_key('Control-Z', None)
            self.viewer.bind_key('Meta-Z', None)

            # Clear drawing state
            self._clear_drawing()

            # Clear references
            self.viewer = None
            self.masks_layer = None
            self._full_masks = None

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _handle_layer_removed(self, event):
        """Handle layer removal events."""
        try:
            layer = event.value
            logger.debug(f"CellCorrection: Layer removed: {layer.name}")

            if layer == self.masks_layer:
                logger.debug("CellCorrection: Masks layer was removed")
                # Only clear reference if it's actually gone from viewer
                if layer not in self.viewer.layers:
                    self.masks_layer = None
                    # Try to recover from visualization manager
                    if self.vis_manager.tracking_layer is not None:
                        logger.debug("CellCorrection: Recovering masks layer from visualization manager")
                        self.masks_layer = self.vis_manager.tracking_layer

            logger.debug(f"CellCorrection: After removal - masks_layer exists: {self.masks_layer is not None}")
            if self.masks_layer is not None:
                logger.debug(f"CellCorrection: After removal - masks_layer in viewer: {self.masks_layer in self.viewer.layers}")

        except Exception as e:
            logger.error(f"CellCorrection: Error handling layer removal: {e}")
