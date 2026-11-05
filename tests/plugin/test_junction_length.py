import numpy as np

from napari_tissuekit.cell_properties import junction_lengths


def _stack():
    """Cells 1 (left) and 2 (right). They touch over 10 rows in frames 2-4 (a 10 px junction), are apart in 0-1 and 5,
    and cell 2 is missing in frame 6."""
    T, H, W = 7, 40, 40
    s = np.zeros((T, H, W), dtype=np.uint16)
    for t in range(T):
        s[t, 10:30, 5:20] = 1
        s[t, 10:30, 20:35] = 2
    for t in (0, 1, 5):
        s[t, 10:30, 20:35] = 0
        s[t, 10:30, 28:35] = 2  # 8 px gap: further apart than the default growing of 3 px
    s[6][s[6] == 2] = 0
    return s


def test_junction_is_measured_only_while_cells_touch():
    df = junction_lengths(_stack(), 1, 2)
    assert list(df.frame) == [2, 3, 4]
    # the T1 tab's measure: growing the cells by 3 px lets the contact run past the rectangle corners
    # (here 3 + 3 px more, because nothing else is around); without growing it is exactly the 20 rows
    assert (df.contact_px == 26).all()
    assert (junction_lengths(_stack(), 1, 2, expand=0).contact_px == 20).all()


def test_order_of_the_two_cells_does_not_matter():
    assert junction_lengths(_stack(), 2, 1).equals(junction_lengths(_stack(), 1, 2))


def test_gap_is_bridged_by_expand():
    df = junction_lengths(_stack(), 1, 2, expand=5)  # 8 px gap, each cell grows 5 px
    assert 0 in set(df.frame)
    assert len(junction_lengths(_stack(), 1, 2, expand=0)) == 3


def test_cells_that_never_touch():
    s = _stack()
    s[:, :, 20:28] = 0
    assert junction_lengths(s, 1, 2, expand=0).empty
