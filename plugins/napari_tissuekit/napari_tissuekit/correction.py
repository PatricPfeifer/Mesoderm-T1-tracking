"""Delete a cell with Option + left click, undo with Cmd/Ctrl+Z, on the "Segmentation" Labels layer.

Reduced from napariCellFlow (Artur Ruppel)'s ``CellCorrectionWidget`` (which also drew new cells), with a slimmed
``DataManager`` and ``VisualizationManager`` that keep only what the widget uses (the stack bookkeeping and the
"Segmentation" Labels layer). napariCellFlow is no longer needed. Licence of the original: see its LICENCE.md /
pyproject.toml.
"""
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Optional

import napari
import numpy as np
from napari.layers import Labels
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

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
    DELETE = "delete"
    BOND = "bond"  # an Enter (apply bond edits) in the outline editor


@dataclass
class UndoAction:
    """One undoable step: ``restore()`` puts things back."""
    action_type: ActionType
    frame: Optional[int]  # Frame number for 3D stacks, None for 2D
    description: str  # Description of the action
    affected_cell_ids: set  # Set of cell IDs affected by this action
    restore: Optional[Callable[[], None]] = None  # undoes the action


class CellCorrectionWidget(QWidget):
    """Option + left click on a cell deletes it; Cmd/Ctrl+Z undoes (also the plugin's bond edits)."""

    correction_made = Signal(np.ndarray)  # Emitted when masks are modified
    MAX_UNDO_STEPS = 20
    CLICK_THRESHOLD = 4  # px: a press that moves less than this is a click
    # Never adopted (renamed to "Segmentation") as the masks layer. "Hovered cell" / "Selected cells" are the
    # Cell properties overlays; renaming them broke cell selection with KeyError "'Hovered cell' is not in list".
    OWN_LAYERS = frozenset({"Outlines", "Outlines (file)", "Tracked", "Hovered cell", "Selected cells"})

    def __init__(self, viewer: "napari.Viewer", data_manager: "DataManager",
                 visualization_manager: "VisualizationManager", parent: QWidget = None):
        super().__init__(parent)
        self.viewer = viewer
        self.data_manager = data_manager
        self.vis_manager = visualization_manager

        self._updating = False
        self._undo_in_progress = False
        self.masks_layer = self.vis_manager.tracking_layer
        self.next_cell_id = 1
        self._full_stack = None
        if self.masks_layer is not None and self.masks_layer.data is not None:
            self.next_cell_id = int(np.max(self.masks_layer.data)) + 1
            self._update_full_stack(self.masks_layer.data)

        self.viewer.layers.events.inserted.connect(self._handle_layer_added)
        self.viewer.layers.events.removed.connect(self._handle_layer_removal)

        self.edit_active = False  # Option is held
        self._saved_mode = None  # (Outlines layer, its tool) while Option is held

        self.undo_stack = deque(maxlen=self.MAX_UNDO_STEPS)
        # Called with the frame before a cell is deleted; returns a function that puts back what it saved (undo).
        self.snapshot_hook: Optional[Callable[[int], Callable[[], None]]] = None
        # Cmd/Ctrl+Z is handled by the plugin panel (TissueKitWidget._undo_edit), which calls undo_last_action;
        # these bindings on the napari viewer remain for use without the panel.
        self.viewer.bind_key('Control-Z', self.undo_last_action)
        self.viewer.bind_key('Meta-Z', self.undo_last_action)

        self._setup_ui()
        if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_drag)
        QApplication.instance().installEventFilter(self)

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        text = QLabel("Option + left click on a cell deletes it.\nCmd+Z / Ctrl+Z undoes the last action.")
        text.setWordWrap(True)
        text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(text)

    # ---- Option held: delete the cell under a click ---------------------------------------------------------
    def eventFilter(self, watched_object, event):
        """Application-wide, so the key is seen whichever widget has the focus."""
        kind = event.type()
        try:
            if kind in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() == Qt.Key_Alt:
                self._set_edit_active(kind == QEvent.KeyPress)
            elif kind in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate):
                self._set_edit_active(False)  # the release would be missed: do not stay on
        except Exception as e:
            logger.error(f"CellCorrection: Error handling the edit key: {e}")
        return False

    def _set_edit_active(self, on: bool) -> None:
        if on == self.edit_active:
            return
        self.edit_active = on
        # The Outlines layer is in paint mode and would paint a dot under the same click: park its tool.
        if on:
            for layer in self.viewer.layers:
                if layer.name == "Outlines" and isinstance(layer, Labels) and layer.mode in ("paint", "erase", "fill"):
                    self._saved_mode = (layer, layer.mode)
                    layer.mode = "pan_zoom"
                    break
        elif self._saved_mode is not None:
            layer, mode = self._saved_mode
            self._saved_mode = None
            if layer in self.viewer.layers:
                layer.mode = mode

    def _on_mouse_drag(self, viewer, event):
        """Option + left click (press and release without moving) deletes the cell under the cursor."""
        try:
            if self.masks_layer is None and self.vis_manager.tracking_layer is not None:
                self.masks_layer = self.vis_manager.tracking_layer
            if self.masks_layer is None or not self.edit_active:
                return
            if event.button not in (1, Qt.LeftButton):
                return
            press = np.round(viewer.cursor.position).astype(int)[-2:]
            yield
            moved = False
            while event.type == 'mouse_move':
                here = np.round(viewer.cursor.position).astype(int)[-2:]
                if np.linalg.norm(here.astype(float) - press.astype(float)) >= self.CLICK_THRESHOLD:
                    moved = True
                yield
            if not moved:
                self._delete_cell_at_position(press)
        except Exception as e:
            logger.error(f"CellCorrection: Error in mouse drag: {e}", exc_info=True)

    def _delete_cell_at_position(self, coords):
        """Delete the cell at the given coordinates (current frame)."""
        if self._updating or not self._validate_coords(coords):
            return
        current_mask = self._get_current_frame_mask()
        if current_mask is None:
            return
        cell_id = int(current_mask[coords[0], coords[1]])
        if cell_id <= 0:
            return
        frame = int(self.viewer.dims.point[0]) if len(self.masks_layer.data.shape) == 3 else 0
        before = current_mask.copy()  # this frame only: the undo step does not copy the whole movie
        # state kept beside the masks (the plugin's outline editor: bonds / outlines of this frame), put back first
        restore_extra = self.snapshot_hook(frame) if self.snapshot_hook is not None else None

        def restore() -> None:
            if restore_extra is not None:
                restore_extra()
            self.data_manager.segmentation_data = (before, frame)
            self.vis_manager.update_tracking_visualization((before, frame))

        try:
            self._updating = True
            new_frame = current_mask.copy()
            new_frame[new_frame == cell_id] = 0
            self.data_manager.segmentation_data = (new_frame, frame)
            self.vis_manager.update_tracking_visualization((new_frame, frame))
            self.undo_stack.append(UndoAction(
                action_type=ActionType.DELETE, frame=frame, description=f"Delete cell {cell_id}",
                affected_cell_ids={cell_id}, restore=restore,
            ))
            self.viewer.status = f"Deleted cell {cell_id}"
        finally:
            self._updating = False

    def undo_last_action(self, event=None):
        """Undo the last step (a cell deletion or a bond edit)."""
        if not self.undo_stack:
            return
        try:
            self._undo_in_progress = True
            self._updating = True
            action = self.undo_stack.pop()
            if action.restore is not None:
                action.restore()
            self.viewer.status = f"Undid: {action.description}"
        except Exception as e:
            logger.error(f"Error during undo: {e}")
            raise
        finally:
            self._undo_in_progress = False
            self._updating = False

    # ---- layers -----------------------------------------------------------------------------------------------
    def _update_full_stack(self, data: np.ndarray):
        """Update the internal full stack reference"""
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        self._full_stack = data.copy()

    def _handle_layer_added(self, event):
        """Handle when a new layer is added to the viewer"""
        if self._updating:
            return
        layer = event.value
        # The plugin's own Labels layers (outlines, tracks) are not masks: renaming them to "Segmentation" hid the
        # Outlines layer from the plugin and made the edit tools work on the wrong layer.
        if layer.name.split(" [")[0] in self.OWN_LAYERS:
            return
        # "Seg · <run>": read-only overlays of other segmentation runs, never the masks being edited.
        if layer.name.startswith("Seg · ") or layer.metadata.get("segmentation_run"):
            return
        if isinstance(layer, napari.layers.Labels):
            try:
                self._updating = True
                self.masks_layer = layer
                layer.name = 'Segmentation'
                data = layer.data
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                if not self.data_manager._initialized:
                    self.data_manager.initialize_stack(data.shape[0])
                self.data_manager.segmentation_data = data
                self.vis_manager.tracking_layer = layer
                self.next_cell_id = int(data.max()) + 1
            except Exception as e:
                logger.error(f"Failed to initialize with external layer: {e}", exc_info=True)
            finally:
                self._updating = False

    def _handle_layer_removal(self, event):
        """Handle layer removal events"""
        if event.value == self.masks_layer:
            self.masks_layer = None

    def _validate_coords(self, coords):
        """Validate coordinates are within image bounds."""
        if self.masks_layer is None:
            return False
        try:
            shape = self.masks_layer.data.shape[1:] if len(self.masks_layer.data.shape) == 3 else self.masks_layer.data.shape
            return 0 <= coords[0] < shape[0] and 0 <= coords[1] < shape[1]
        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return False

    def set_masks_layer(self, masks: np.ndarray):
        """Set or update the masks layer."""
        if self._updating:
            return
        try:
            self._updating = True
            if masks.ndim == 2:
                masks = masks[np.newaxis, ...]
            self.vis_manager.update_tracking_visualization(masks)
            self.masks_layer = self.vis_manager.tracking_layer
            self.vis_manager.tracking_layer = self.masks_layer
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
            return self.masks_layer.data[int(self.viewer.dims.point[0])]
        return self.masks_layer.data

    def cleanup(self):
        """Clean up resources and disconnect events"""
        try:
            self._set_edit_active(False)
            QApplication.instance().removeEventFilter(self)
            self.viewer.layers.events.inserted.disconnect(self._handle_layer_added)
            self.viewer.layers.events.removed.disconnect(self._handle_layer_removal)
            if self._on_mouse_drag in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
            self.viewer.bind_key('Control-Z', None)
            self.viewer.bind_key('Meta-Z', None)
            self.viewer = None
            self.masks_layer = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
