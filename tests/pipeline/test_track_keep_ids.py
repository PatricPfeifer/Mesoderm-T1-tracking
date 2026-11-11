"""Keep list: cells whose ID no track edit may change (not even as a side effect of Link's exchange)."""
import numpy as np
import pytest

from napari_tissuekit.track_edit import ProtectedCellsError, break_track, link_tracks, swap_tracks


def stack():
    """Two cells side by side, 4 frames. After frame 1 their IDs are crossed by a tracking error:
    cell B (left) is 2 on frames 0-1 and 7 on frames 2-3; cell A (right) is 1 on frames 0-1 and 2 on frames 2-3.
    A third cell C (id 5) is a bystander."""
    s = np.zeros((4, 30, 60), np.int32)
    for t in range(4):
        s[t, 2:12, 2:12] = 2 if t < 2 else 7  # B
        s[t, 2:12, 20:30] = 1 if t < 2 else 2  # A
        s[t, 18:28, 2:12] = 5  # C
    return s


def test_link_exchange_is_refused_when_the_other_cell_is_kept():
    s = stack()
    before = s.copy()
    with pytest.raises(ProtectedCellsError) as e:
        link_tracks(s, 2, 7, 2, keep_ids={2})  # B takes ID 2 back, which would turn A (kept, ID 2) into 7
    assert "keep list" in str(e.value) and e.value.frame == 2 and len(e.value.ys) == 100
    assert (s == before).all()


def test_link_exchange_goes_through_when_nothing_relevant_is_kept():
    s = stack()
    undos, msg = link_tracks(s, 2, 7, 2, keep_ids={5})  # only the bystander is kept: the exchange may happen
    assert undos and (s[2:, 2:12, 2:12] == 2).all() and (s[2:, 2:12, 20:30] == 7).all() and (s[:, 18:28, 2:12] == 5).all()


def test_link_refused_when_the_clicked_cell_itself_is_kept():
    s = stack()
    with pytest.raises(ProtectedCellsError):
        link_tracks(s, 2, 7, 2, keep_ids={7})


def test_break_and_swap_refuse_kept_cells():
    s = stack()
    with pytest.raises(ProtectedCellsError):
        break_track(s, 5, 1, keep_ids={5})
    with pytest.raises(ProtectedCellsError):
        swap_tracks(s, 5, 7, 2, keep_ids={5})
    undos, _ = break_track(s, 5, 1, keep_ids={7})  # a different cell is kept: fine
    assert undos


def test_empty_keep_list_changes_nothing():
    a, b = stack(), stack()
    link_tracks(a, 2, 7, 2)
    link_tracks(b, 2, 7, 2, keep_ids=set())
    assert (a == b).all()
