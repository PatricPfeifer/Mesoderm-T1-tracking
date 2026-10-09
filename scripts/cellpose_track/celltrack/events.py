"""Event classification: continue / divide / appear / disappear / merge."""

from __future__ import annotations

from dataclasses import dataclass

from .assign import Assignment
from .cells import FrameCells


@dataclass
class Event:
    kind: str  # continue | divide | appear | disappear | merge
    frame_from: int  # original on-disk frame index
    frame_to: int
    label_from: int | None
    label_to: int | None
    track_id: int | None = None
    track_id_b: int | None = None  # second daughter / merge partner
    cost: float | None = None
    topology_score: float | None = None
    flagged: bool = False
    note: str = ""


def classify_events(
    assignment: Assignment,
    prev: FrameCells,
    cur: FrameCells,
    prev_label_to_track: dict[int, int],
    cur_label_to_track: dict[int, int],
    per_track_score: dict[int, float] | None = None,
) -> list[Event]:
    """Classify each link / unmatched cell into an Event."""
    per_track_score = per_track_score or {}
    events: list[Event] = []
    f0 = prev.original_frame
    f1 = cur.original_frame

    by_parent: dict[int, list[int]] = {}
    for p, c in assignment.matches:
        by_parent.setdefault(p, []).append(c)

    for p, kids in by_parent.items():
        prev_lab = int(prev.original_labels[p + 1])
        if len(kids) == 1:
            c = kids[0]
            cur_lab = int(cur.original_labels[c + 1])
            track = cur_label_to_track.get(cur_lab)
            events.append(
                Event(
                    kind="continue",
                    frame_from=f0,
                    frame_to=f1,
                    label_from=prev_lab,
                    label_to=cur_lab,
                    track_id=track,
                    cost=assignment.costs.get((p, c)),
                    topology_score=per_track_score.get(track) if track else None,
                )
            )
        else:
            daughter_labs = [int(cur.original_labels[c + 1]) for c in kids]
            daughter_tracks = [cur_label_to_track.get(lab) for lab in daughter_labs]
            events.append(
                Event(
                    kind="divide",
                    frame_from=f0,
                    frame_to=f1,
                    label_from=prev_lab,
                    label_to=daughter_labs[0],
                    track_id=daughter_tracks[0],
                    track_id_b=daughter_tracks[1] if len(daughter_tracks) > 1 else None,
                    cost=min(
                        assignment.costs.get((p, c), 0.0) for c in kids
                    ),
                    note=f"daughters={daughter_labs}",
                )
            )

    for p in assignment.disappeared:
        prev_lab = int(prev.original_labels[p + 1])
        track = prev_label_to_track.get(prev_lab)
        events.append(
            Event(
                kind="disappear",
                frame_from=f0,
                frame_to=f1,
                label_from=prev_lab,
                label_to=None,
                track_id=track,
            )
        )

    for j in assignment.appeared:
        skip = any(len(kids) > 1 and j in kids for kids in by_parent.values())
        if skip:
            continue
        cur_lab = int(cur.original_labels[j + 1])
        track = cur_label_to_track.get(cur_lab)
        events.append(
            Event(
                kind="appear",
                frame_from=f0,
                frame_to=f1,
                label_from=None,
                label_to=cur_lab,
                track_id=track,
            )
        )

    return events
