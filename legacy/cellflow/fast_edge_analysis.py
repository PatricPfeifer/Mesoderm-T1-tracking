"""Standalone speed copy of napariCellFlow EdgeAnalyzer.

Vendor plugin and scripts/cellflow/run_cellflow_t1.py are not imported or edited.
T1 validation and edge-identity tracking match the original; only candidate
pair finding and shared-boundary morphology are faster.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import cv2
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import expand_labels

logger = logging.getLogger(__name__)


@dataclass
class EdgeAnalysisParams:
    dilation_radius: int = 1
    min_overlap_pixels: int = 5
    min_edge_length: float = 0.0
    filter_isolated: bool = True
    temporal_window: int = 1
    min_contact_frames: int = 1

    def validate(self) -> None:
        if self.dilation_radius < 1 or self.dilation_radius > 10:
            raise ValueError("Dilation radius must be between 1 and 10")
        if self.min_overlap_pixels < 1 or self.min_overlap_pixels > 100:
            raise ValueError("Minimum overlap pixels must be between 1 and 100")
        if self.min_edge_length < 0:
            raise ValueError("Minimum edge length cannot be negative")
        if self.temporal_window < 1:
            raise ValueError("Temporal window must be positive")
        if self.min_contact_frames < 1:
            raise ValueError("Minimum contact frames must be positive")
        if self.min_contact_frames > self.temporal_window:
            raise ValueError("Minimum contact frames cannot exceed temporal window")


@dataclass
class CellBoundary:
    cell_ids: Tuple[int, int]
    coordinates: np.ndarray
    endpoint1: np.ndarray
    endpoint2: np.ndarray
    length: float


@dataclass
class IntercalationEvent:
    frame: int
    losing_cells: Tuple[int, int]
    gaining_cells: Tuple[int, int]
    coordinates: np.ndarray


@dataclass
class EdgeData:
    edge_id: int
    frames: List[int]
    cell_pairs: List[Tuple[int, int]]
    lengths: List[float]
    coordinates: List[np.ndarray]
    time_points: List[float] = field(default_factory=list)
    intercalations: List["IntercalationEvent"] = field(default_factory=list)

    def add_frame(self, frame: int, cells: Tuple[int, int], length: float, coords: np.ndarray) -> None:
        self.frames.append(frame)
        self.cell_pairs.append(cells)
        self.lengths.append(length)
        self.coordinates.append(coords)

    def add_intercalation(self, event: IntercalationEvent) -> None:
        self.intercalations.append(event)
        self.intercalations.sort(key=lambda x: x.frame)


@dataclass
class EdgeAnalysisResults:
    def __init__(self, params: EdgeAnalysisParams):
        self.edges: Dict[int, EdgeData] = {}
        self.segmentation_data: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {
            "total_frames": 0,
            "frame_ids": [],
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
        }

    def add_edge(self, edge: EdgeData) -> None:
        self.edges[edge.edge_id] = edge

    def get_edge(self, edge_id: int) -> Optional[EdgeData]:
        return self.edges.get(edge_id)

    def set_segmentation_data(self, data: np.ndarray) -> None:
        self.segmentation_data = data

    def get_segmentation_data(self) -> Optional[np.ndarray]:
        return self.segmentation_data

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def validate(self) -> None:
        if not self.edges:
            raise ValueError("No edge data present")
        if not self.metadata["frame_ids"]:
            raise ValueError("No frames recorded")
        if not isinstance(self.metadata["parameters"], EdgeAnalysisParams):
            raise ValueError("Invalid or missing analysis parameters")
        if self.segmentation_data is None:
            raise ValueError("No segmentation data present")


@dataclass
class EdgeGroup:
    group_id: int
    edge_ids: Set[int] = field(default_factory=set)
    cell_pairs: Set[FrozenSet[int]] = field(default_factory=set)
    frames: List[int] = field(default_factory=list)
    active: bool = True


def _adjacent_pairs(labels: np.ndarray, expand: int) -> List[Tuple[int, int]]:
    """Touching label pairs after expand_labels (4-neighborhood of shifts)."""
    work = expand_labels(labels, distance=expand) if expand > 0 else labels
    valid = {int(v) for v in np.unique(labels) if v != 0}
    height, width = work.shape
    pairs: Set[Tuple[int, int]] = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = work[: height - dy, max(0, -dx) : width - max(0, dx)]
        b = work[dy:, max(0, dx) : width + min(0, dx)]
        mask = (a > 0) & (b > 0) & (a != b)
        if not mask.any():
            continue
        for u, v in zip(a[mask].ravel().tolist(), b[mask].ravel().tolist()):
            u, v = int(u), int(v)
            if u in valid and v in valid:
                pairs.add((u, v) if u < v else (v, u))
    return list(pairs)


def _bboxes_too_far(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    max_gap: int,
) -> bool:
    """True if axis-aligned boxes cannot overlap after max_gap dilation.

    box is (minr, minc, maxr, maxc) as from skimage regionprops (max exclusive).
    """
    r1a, c1a, r1b, c1b = box1
    r2a, c2a, r2b, c2b = box2
    if r1b + max_gap < r2a or r2b + max_gap < r1a:
        return True
    if c1b + max_gap < c2a or c2b + max_gap < c1a:
        return True
    return False


class FastEdgeAnalyzer:
    """Copy of napariCellFlow EdgeAnalyzer with faster edge detection."""

    def __init__(self, params: Optional[EdgeAnalysisParams] = None):
        self.params = params or EdgeAnalysisParams()
        self.next_edge_id = 1
        self.next_group_id = 1
        self._edge_history = {}
        self._edge_groups = {}
        self._cell_pair_to_edge_id = {}
        self._edge_to_group = {}
        self._frame_graphs = {}

    def _normalize_cell_pair(
        self, cells: Union[Tuple[int, int], List[int], np.ndarray]
    ) -> Tuple[Union[FrozenSet[int], Tuple[int, int]], Tuple[int, int]]:
        if isinstance(cells, (tuple, list, np.ndarray)) and len(cells) == 2:
            ordered = tuple(sorted(int(c) for c in cells))
            frozen = frozenset(ordered)
            return frozen, ordered
        raise ValueError("Cell pair must be a tuple/list/array of exactly 2 cells")

    def _create_edge_data(self, edge_id: int, frame: int, boundary: CellBoundary) -> EdgeData:
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        return EdgeData(
            edge_id=edge_id,
            frames=[frame],
            cell_pairs=[ordered_pair],
            lengths=[boundary.length],
            coordinates=[boundary.coordinates],
        )

    def _update_edge_data(self, edge_data: EdgeData, frame: int, boundary: CellBoundary) -> None:
        _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
        edge_data.frames.append(frame)
        edge_data.cell_pairs.append(ordered_pair)
        edge_data.lengths.append(boundary.length)
        edge_data.coordinates.append(boundary.coordinates)

    def _update_edge_histories(self, frame: int, boundaries: List[CellBoundary]):
        for boundary in boundaries:
            frozen_pair, _ = self._normalize_cell_pair(boundary.cell_ids)
            edge_id = self._get_or_create_edge_id(frozen_pair)
            if edge_id in self._edge_history:
                self._update_edge_data(self._edge_history[edge_id], frame, boundary)
            else:
                self._edge_history[edge_id] = self._create_edge_data(edge_id, frame, boundary)

    def _create_frame_graph(self, boundaries: List[CellBoundary]) -> nx.Graph:
        G = nx.Graph()
        for boundary in boundaries:
            _, ordered_pair = self._normalize_cell_pair(boundary.cell_ids)
            G.add_edge(*ordered_pair, boundary=boundary)
        return G

    def _get_or_create_edge_id(self, frozen_pair: FrozenSet[int]) -> int:
        if frozen_pair in self._cell_pair_to_edge_id:
            return self._cell_pair_to_edge_id[frozen_pair]
        edge_id = self.next_edge_id
        self.next_edge_id += 1
        self._cell_pair_to_edge_id[frozen_pair] = edge_id
        return edge_id

    def _validate_t1_transition(
        self,
        lost_edge: FrozenSet[int],
        gained_edge: FrozenSet[int],
        G1: nx.Graph,
        G2: nx.Graph,
    ) -> bool:
        all_cells = lost_edge | gained_edge
        if len(all_cells) != 4:
            return False
        lost_cells = tuple(sorted(lost_edge))
        gained_cells = tuple(sorted(gained_edge))
        edges_to_check = [
            tuple(sorted((lost_cells[0], gained_cells[0]))),
            tuple(sorted((lost_cells[0], gained_cells[1]))),
            tuple(sorted((lost_cells[1], gained_cells[0]))),
            tuple(sorted((lost_cells[1], gained_cells[1]))),
        ]
        edges_t = set(tuple(sorted(e)) for e in G1.edges())
        edges_t_plus_1 = set(tuple(sorted(e)) for e in G2.edges())
        connecting_edges_exist = all(
            (e in edges_t and e in edges_t_plus_1)
            for e in edges_to_check
            if e != lost_cells and e != gained_cells
        )
        return connecting_edges_exist

    def _create_edge_trajectories(
        self,
        boundaries_by_frame: Dict[int, List[CellBoundary]],
        intercalations: List[IntercalationEvent],
    ) -> Dict[int, EdgeData]:
        logger.info("Creating edge trajectories with forward-time merging...")
        sorted_events = sorted(intercalations, key=lambda x: x.frame)
        edge_trajectories = {}
        cell_pair_to_trajectory = {}
        next_trajectory_id = 1
        logger.debug(f"Processing {len(sorted_events)} intercalation events")
        for event in sorted_events:
            losing_pair = frozenset(event.losing_cells)
            gaining_pair = frozenset(event.gaining_cells)
            losing_traj_id = cell_pair_to_trajectory.get(losing_pair)
            if losing_traj_id is None:
                losing_traj_id = next_trajectory_id
                next_trajectory_id += 1
                edge_trajectories[losing_traj_id] = EdgeData(
                    edge_id=losing_traj_id,
                    frames=[],
                    cell_pairs=[],
                    lengths=[],
                    coordinates=[],
                    intercalations=[event],
                )
                cell_pair_to_trajectory[losing_pair] = losing_traj_id
            else:
                edge_trajectories[losing_traj_id].intercalations.append(event)
            cell_pair_to_trajectory[gaining_pair] = losing_traj_id
            logger.debug(
                f"Merged trajectory {losing_traj_id}: {event.losing_cells} → {event.gaining_cells}"
            )
        logger.debug("Processing boundaries frame by frame")
        for frame, boundaries in sorted(boundaries_by_frame.items()):
            for boundary in boundaries:
                cell_pair = frozenset(boundary.cell_ids)
                if cell_pair not in cell_pair_to_trajectory:
                    traj_id = next_trajectory_id
                    next_trajectory_id += 1
                    cell_pair_to_trajectory[cell_pair] = traj_id
                    edge_trajectories[traj_id] = EdgeData(
                        edge_id=traj_id,
                        frames=[],
                        cell_pairs=[],
                        lengths=[],
                        coordinates=[],
                        intercalations=[],
                    )
                traj_id = cell_pair_to_trajectory[cell_pair]
                trajectory = edge_trajectories[traj_id]
                sign = -1
                for event in trajectory.intercalations:
                    if frame > event.frame:
                        sign *= -1
                trajectory.frames.append(frame)
                trajectory.cell_pairs.append(tuple(sorted(boundary.cell_ids)))
                trajectory.lengths.append(sign * boundary.length)
                trajectory.coordinates.append(boundary.coordinates)
        intercalating_edges = sum(1 for t in edge_trajectories.values() if t.intercalations)
        total_intercalations = sum(len(t.intercalations) for t in edge_trajectories.values())
        logger.info(f"Created {len(edge_trajectories)} edge trajectories")
        logger.debug(f"Found {intercalating_edges} edges involved in intercalations")
        logger.debug(f"Total intercalation events processed: {total_intercalations}")
        return edge_trajectories

    def _merge_edge_groups(self, lost_edge: FrozenSet[int], gained_edge: FrozenSet[int], frame: int):
        related_groups = set()
        for group in self._edge_groups.values():
            if lost_edge in group.cell_pairs or gained_edge in group.cell_pairs:
                related_groups.add(group.group_id)
                for cell_pair in group.cell_pairs:
                    for other_group in self._edge_groups.values():
                        if cell_pair in other_group.cell_pairs:
                            related_groups.add(other_group.group_id)
        if not related_groups:
            group = EdgeGroup(self.next_group_id)
            self.next_group_id += 1
            group.edge_ids.update(
                {self._cell_pair_to_edge_id[lost_edge], self._cell_pair_to_edge_id[gained_edge]}
            )
            group.cell_pairs.update({lost_edge, gained_edge})
            group.frames.append(frame)
            self._edge_groups[group.group_id] = group
        else:
            base_group_id = min(related_groups)
            merged = EdgeGroup(base_group_id)
            for group_id in related_groups:
                group = self._edge_groups[group_id]
                merged.edge_ids.update(group.edge_ids)
                merged.cell_pairs.update(group.cell_pairs)
                merged.frames.extend(group.frames)
                if group_id != base_group_id:
                    del self._edge_groups[group_id]
            merged.edge_ids.update(
                {self._cell_pair_to_edge_id[lost_edge], self._cell_pair_to_edge_id[gained_edge]}
            )
            merged.cell_pairs.update({lost_edge, gained_edge})
            merged.frames.append(frame)
            merged.frames = sorted(set(merged.frames))
            self._edge_groups[base_group_id] = merged
            for edge_id in merged.edge_ids:
                self._edge_to_group[edge_id] = base_group_id

    def _detect_edges(self, frame_data: np.ndarray) -> List[CellBoundary]:
        expand = max(int(self.params.dilation_radius) + 1, 2)
        bboxes = {int(p.label): p.bbox for p in regionprops(frame_data)}
        pairs = _adjacent_pairs(frame_data, expand)
        boundaries = []
        for cell1, cell2 in pairs:
            boundary = self._find_shared_boundary(
                frame_data, cell1, cell2, bboxes.get(int(cell1)), bboxes.get(int(cell2))
            )
            if boundary is not None:
                boundaries.append(boundary)
        if self.params.filter_isolated:
            boundaries = self._filter_isolated_edges(boundaries)
        return boundaries

    def _filter_isolated_edges(self, boundaries: List[CellBoundary]) -> List[CellBoundary]:
        connections = defaultdict(set)
        for boundary in boundaries:
            cell1, cell2 = boundary.cell_ids
            connections[int(cell1)].add(int(cell2))
            connections[int(cell2)].add(int(cell1))
        return [
            b
            for b in boundaries
            if len(connections[int(b.cell_ids[0])]) > 1 and len(connections[int(b.cell_ids[1])]) > 1
        ]

    def update_parameters(self, params: EdgeAnalysisParams) -> None:
        self.params = params
        self._edge_history.clear()
        self._edge_groups.clear()
        self._cell_pair_to_edge_id.clear()
        self._edge_to_group.clear()
        self._frame_graphs.clear()
        self.next_edge_id = 1
        self.next_group_id = 1

    def _find_shared_boundary(
        self,
        frame: np.ndarray,
        cell1_id: int,
        cell2_id: int,
        box1: Optional[Tuple[int, int, int, int]] = None,
        box2: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[CellBoundary]:
        radius = int(self.params.dilation_radius)
        max_gap = 2 * radius
        if box1 is not None and box2 is not None and _bboxes_too_far(box1, box2, max_gap):
            return None

        h, w = frame.shape
        pad = radius + 2
        if box1 is not None and box2 is not None:
            minr = min(box1[0], box2[0])
            minc = min(box1[1], box2[1])
            maxr = max(box1[2], box2[2])
            maxc = max(box1[3], box2[3])
            y0 = max(0, minr - pad)
            x0 = max(0, minc - pad)
            y1 = min(h, maxr + pad)
            x1 = min(w, maxc + pad)
            crop = frame[y0:y1, x0:x1]
        else:
            y0, x0 = 0, 0
            crop = frame

        mask1 = (crop == cell1_id).astype(np.uint8)
        mask2 = (crop == cell2_id).astype(np.uint8)
        if not mask1.any() or not mask2.any():
            return None

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        dilated1 = cv2.dilate(mask1, kernel)
        dilated2 = cv2.dilate(mask2, kernel)
        overlap = dilated1 & dilated2
        if not np.any(overlap):
            return None

        thin_kernel = np.ones((3, 3), np.uint8)
        thin_dilated1 = cv2.dilate(mask1, thin_kernel)
        thin_dilated2 = cv2.dilate(mask2, thin_kernel)
        boundary_region = thin_dilated1 & thin_dilated2
        if np.sum(boundary_region) < self.params.min_overlap_pixels:
            return None

        skeleton = skeletonize(boundary_region)
        if not np.any(skeleton):
            return None

        ordered_coords = self._order_boundary_pixels(skeleton)
        if len(ordered_coords) < 2:
            return None
        ordered_coords = ordered_coords.astype(np.int64, copy=False)
        ordered_coords = ordered_coords + np.array([y0, x0], dtype=np.int64)

        length = self._calculate_edge_length(ordered_coords)
        if self.params.min_edge_length > 0 and length < self.params.min_edge_length:
            return None

        return CellBoundary(
            cell_ids=(cell1_id, cell2_id),
            coordinates=ordered_coords,
            endpoint1=ordered_coords[0],
            endpoint2=ordered_coords[-1],
            length=length,
        )

    def _order_boundary_pixels(self, skeleton: np.ndarray) -> np.ndarray:
        points = np.column_stack(np.where(skeleton))
        if len(points) <= 2:
            return points
        endpoint_indices = []
        for i, point in enumerate(points):
            y, x = point
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx]:
                            neighbors += 1
            if neighbors == 1:
                endpoint_indices.append(i)
        if len(endpoint_indices) != 2:
            distances = cdist(points, points)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            endpoint_indices = [i, j]
        ordered = [points[endpoint_indices[0]]]
        remaining = list(range(len(points)))
        remaining.remove(endpoint_indices[0])
        while remaining:
            current = ordered[-1]
            min_dist = float("inf")
            next_idx = None
            for idx in remaining:
                dist = np.sum((points[idx] - current) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    next_idx = idx
            if next_idx is None:
                break
            ordered.append(points[next_idx])
            remaining.remove(next_idx)
        return np.array(ordered)

    def _calculate_edge_length(self, coords: np.ndarray) -> float:
        if len(coords) < 2:
            return 0.0
        diffs = np.diff(coords, axis=0)
        segments = np.sqrt(np.sum(diffs ** 2, axis=1))
        return float(np.sum(segments))

    def _detect_topology_changes(self, frame: int, next_frame: int) -> List[IntercalationEvent]:
        G1 = self._frame_graphs[frame]
        G2 = self._frame_graphs[next_frame]
        edges_t = set(tuple(sorted((u, v))) for u, v in G1.edges())
        edges_t_plus_1 = set(tuple(sorted((u, v))) for u, v in G2.edges())
        removed_edges = edges_t - edges_t_plus_1
        added_edges = edges_t_plus_1 - edges_t
        logger.debug(f"Removed edges: {removed_edges}")
        logger.debug(f"Added edges: {added_edges}")
        events = []
        for removed_edge in removed_edges:
            for added_edge in added_edges:
                if self._validate_t1_transition(
                    frozenset(removed_edge), frozenset(added_edge), G1, G2
                ):
                    boundary = G1.edges[removed_edge]["boundary"]
                    coords = np.mean(boundary.coordinates, axis=0)
                    lost_pair = frozenset(removed_edge)
                    gained_pair = frozenset(added_edge)
                    if gained_pair not in self._cell_pair_to_edge_id:
                        self._cell_pair_to_edge_id[gained_pair] = self.next_edge_id
                        self.next_edge_id += 1
                    self._merge_edge_groups(lost_pair, gained_pair, frame)
                    event = IntercalationEvent(
                        frame=frame,
                        losing_cells=removed_edge,
                        gaining_cells=added_edge,
                        coordinates=coords,
                    )
                    events.append(event)
                    logger.debug(
                        f"Found intercalation at frame {frame}: {removed_edge} → {added_edge}"
                    )
        logger.debug(f"Found {len(events)} events between frames {frame} and {next_frame}")
        return events

    def analyze_sequence(self, tracked_data: np.ndarray, progress_callback=None) -> EdgeAnalysisResults:
        logger.info("Starting edge analysis sequence...")
        self._edge_history.clear()
        self._edge_groups.clear()
        self._cell_pair_to_edge_id.clear()
        self._edge_to_group.clear()
        self._frame_graphs.clear()
        self.next_edge_id = 1
        self.next_group_id = 1
        results = EdgeAnalysisResults(self.params)
        results.update_metadata("total_frames", len(tracked_data))
        results.update_metadata("frame_ids", list(range(len(tracked_data))))
        results.set_segmentation_data(tracked_data)
        boundaries_by_frame = {}
        all_intercalations = []
        total_frames = len(tracked_data)
        for frame_num in range(total_frames):
            if progress_callback:
                progress = int(100 * frame_num / total_frames)
                progress_callback(progress, f"Processing frame {frame_num + 1}/{total_frames}")
            logger.debug(f"Processing frame {frame_num}")
            boundaries = self._detect_edges(tracked_data[frame_num])
            boundaries_by_frame[frame_num] = boundaries
            self._frame_graphs[frame_num] = self._create_frame_graph(boundaries)
            if frame_num > 0:
                events = self._detect_topology_changes(frame_num - 1, frame_num)
                all_intercalations.extend(events)
            self._update_edge_histories(frame_num, boundaries)
        if progress_callback:
            progress_callback(100, "Finalizing analysis...")
        logger.info(f"Found total of {len(all_intercalations)} intercalations")
        edge_trajectories = self._create_edge_trajectories(boundaries_by_frame, all_intercalations)
        results.edges = edge_trajectories
        results.validate()
        return results
