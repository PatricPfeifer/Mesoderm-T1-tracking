"""T1 preview: finding the clicked event, drawing its frames, event order."""
import numpy as np

from napari_tissuekit.t1_preview import (
    CLICK_RADIUS, event_box, event_frames, find_event_at, render_preview_frame, sort_events,
)


def labels_2x2(n=12, size=30, swap_at=7):
    """Four cells meeting at the centre. Before `swap_at`: A (left) and D (right) share a vertical junction, B on top and C at
    the bottom. From `swap_at`: B (left) and C (right) share the junction, A on top and D at the bottom."""
    lab = np.zeros((n, size, size), np.uint32)
    h = size // 2
    for t in range(n):
        top, bot, left, right = (2, 3, 1, 4) if t < swap_at else (1, 4, 2, 3)
        lab[t, :h - 3, :], lab[t, h + 3:, :] = top, bot
        lab[t, h - 3:h + 3, :h], lab[t, h - 3:h + 3, h:] = left, right
    return lab


def ev(quartet=(1, 2, 3, 4), t0=4, t1=7, swap=5, typ="completed", y=15.0, x=15.0):
    q = frozenset(quartet)
    a, b, c, d = sorted(quartet)
    return dict(quartet=q, A=a, B=b, C=c, D=d, t_start=t0, t_end=t1, t_first_swap=swap, t_last_swap=swap, type=typ, t1_class="clean",
                y=y, x=x, pair_before=frozenset((a, d)), pair_after=frozenset((b, c)), pair_building=f"{b};{c}")


def test_event_frames_window_is_clipped_to_the_movie():
    e = ev()
    assert list(event_frames(e, 3, 3, 12)) == list(range(1, 11))
    assert list(event_frames(e, 10, 10, 12)) == list(range(0, 12))


def test_click_on_a_quartet_cell_or_near_the_vertex_finds_the_event():
    e = ev()
    four = [{e["quartet"]: (15.0, 15.0)} for _ in range(12)]
    assert find_event_at([e], four, 5, 3, 3, 1, 3, 3) is e              # on cell A
    assert find_event_at([e], four, 5, 17, 18, 0, 3, 3) is e            # background pixel near the vertex
    assert find_event_at([e], four, 5, 28, 2, 0, 3, 3) is None          # far from it, not a quartet cell
    assert find_event_at([e], four, 5, 28, 2, 9, 3, 3) is None          # other cell, far away
    assert find_event_at([e], four, 11, 15, 15, 1, 3, 3) is None        # outside the frames the overlay shows
    assert find_event_at([e], four, 10, 15, 15, 1, 3, 3) is e           # last frame of the post window


def test_overlapping_events_nearest_vertex_wins():
    e1, e2 = ev((1, 2, 3, 4), y=10.0, x=10.0), ev((1, 2, 5, 6), y=20.0, x=20.0)
    four = [{e1["quartet"]: (10.0, 10.0), e2["quartet"]: (20.0, 20.0)} for _ in range(12)]
    assert find_event_at([e1, e2], four, 5, 11, 11, 1, 3, 3) is e1      # cell 1 is in both: nearest vertex decides
    assert find_event_at([e1, e2], four, 5, 19, 19, 1, 3, 3) is e2
    assert CLICK_RADIUS > 0


def test_sort_events_like_the_picker():
    a, b, c = ev(typ="flicker_aborted", t0=1), ev(typ="completed", t0=9), ev(typ="flicker_resolved", t0=2)
    assert [e["type"] for e in sort_events([a, b, c])] == ["completed", "flicker_resolved", "flicker_aborted"]


def test_render_has_quartet_colours_junctions_and_vertex_and_crops():
    lab = labels_2x2()
    e = ev()
    box = event_box(e, lab)
    assert 0 <= box[0] < box[1] <= 30 and 0 <= box[2] < box[3] <= 30
    during = render_preview_frame(lab[7], e, 7, None, reach=1, box=box, scale=2)
    h, w = (box[1] - box[0]) * 2, (box[3] - box[2]) * 2
    assert during.shape == (h, w, 3) and during.dtype == np.uint8
    colours = {tuple(px) for px in during.reshape(-1, 3)}
    for want in ((170, 130, 190), (130, 175, 155), (130, 155, 195), (195, 150, 140)):  # the four quartet fills
        assert want in colours
    assert (0, 220, 255) in colours and (0, 180, 220) in colours  # swapped frame: building junction and vertex
    d6 = {tuple(px) for px in render_preview_frame(lab[6], e, 6, None, reach=1, box=box, scale=2).reshape(-1, 3)}
    assert (255, 140, 0) in d6 and (0, 180, 220) in d6  # before the swap, still inside the event: breaking junction and vertex
    before = render_preview_frame(lab[2], e, 2, None, reach=1, box=box, scale=2)
    colours_b = {tuple(px) for px in before.reshape(-1, 3)}
    assert (255, 140, 0) in colours_b and (0, 220, 255) not in colours_b and (0, 180, 220) not in colours_b
    after = render_preview_frame(lab[9], e, 9, None, reach=1, box=box, scale=2)
    colours_a = {tuple(px) for px in after.reshape(-1, 3)}
    assert (0, 220, 255) in colours_a and (255, 140, 0) not in colours_a
