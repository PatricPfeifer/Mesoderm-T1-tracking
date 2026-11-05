"""Segmentation TIFFs are saved losslessly compressed; other outputs are not."""
import numpy as np
import pytest
import tifffile

from napari_tissuekit.core.movie_calibration import Calibration, read_calibration, save_tiff


def _masks():
    rng = np.random.default_rng(0)
    m = np.zeros((3, 64, 64), np.uint16)
    for t in range(3):
        for i, (y, x) in enumerate(rng.integers(0, 56, (20, 2)), start=1):
            m[t, y:y + 8, x:x + 8] = i
    return m


def _compression(path):
    with tifffile.TiffFile(path) as tif:
        return {p.compression for p in tif.pages}


@pytest.mark.parametrize("cal", [None, Calibration(um_per_px_x=0.5, um_per_px_y=0.5, s_per_frame=30.0)])
@pytest.mark.parametrize("name,dtype", [("egg_cp_masks.tif", np.uint16), ("egg_cell_outlines.tif", np.uint8)])
def test_segmentation_outputs_compressed_lossless(tmp_path, cal, name, dtype):
    data = _masks().astype(dtype)
    if dtype == np.uint8:
        data = (data > 0).astype(np.uint8) * 255
    path = tmp_path / name
    save_tiff(path, data, cal)
    assert _compression(path) == {tifffile.COMPRESSION.ADOBE_DEFLATE}
    assert np.array_equal(tifffile.imread(path), data)
    assert path.stat().st_size < data.nbytes / 5
    if cal is not None:
        back = read_calibration(path)
        assert back.um_per_px == pytest.approx(0.5)
        assert back.s_per_frame == pytest.approx(30.0)


def test_other_outputs_uncompressed(tmp_path):
    path = tmp_path / "egg_tracked_labels.tif"
    save_tiff(path, _masks().astype(np.uint32), None)
    assert _compression(path) == {tifffile.COMPRESSION.NONE}


def test_compression_override(tmp_path):
    path = tmp_path / "egg_cp_masks.tif"
    save_tiff(path, _masks(), None, compression=None)
    assert _compression(path) == {tifffile.COMPRESSION.NONE}
