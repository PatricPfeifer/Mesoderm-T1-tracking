"""Frame-range selection in the T1 picker."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "picker", ROOT / "plugins" / "napari_tissuekit" / "napari_tissuekit" / "core" / "t1" / "run_tm_t1_picker.py"
)
picker = importlib.util.module_from_spec(spec)
sys.modules["picker"] = picker
spec.loader.exec_module(picker)


def make_event(first, last, start=None):
    start = first if start is None else start
    return picker.T1Event(
        event_id=1, frame_start=start, frame_end=last,
        frame_first_swap=first, frame_last_swap=last,
        frame_before=first - 1, frame_after=last + 1,
        A=1, B=2, C=3, D=4,
        pair_breaking=frozenset((1, 2)), pair_building=frozenset((3, 4)),
        y=0.0, x=0.0,
    )


def test_event_in_range():
    ev = make_event(5, 7)
    assert picker.event_in_range(ev, None)
    assert picker.event_in_range(ev, (5, 7))
    assert picker.event_in_range(ev, (0, 20))
    assert picker.event_in_range(ev, (0, 5))
    assert picker.event_in_range(ev, (5, 5))
    assert not picker.event_in_range(ev, (6, 20))
    assert not picker.event_in_range(ev, (0, 4))


def test_event_in_range_uses_earlier_of_swap_and_fourway():
    ev = make_event(5, 9, start=7)
    assert picker.event_in_range(ev, (0, 5))
    ev = make_event(8, 9, start=6)
    assert picker.event_in_range(ev, (6, 6))
    assert not picker.event_in_range(ev, (7, 20))


def test_trace_span_ignores_range():
    ev = make_event(10, 12)
    assert picker.length_trace_span(ev, 50, 5) == (5, 17)
    assert picker.length_trace_span(ev, 50, 10) == (0, 22)
    assert picker.length_trace_span(ev, 15, 10) == (0, 14)
