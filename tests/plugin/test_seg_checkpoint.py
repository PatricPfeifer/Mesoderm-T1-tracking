"""``tissuekit segment`` writes its resume file every N frames (not every frame), and on SIGTERM."""
import os
import signal

import numpy as np
import pytest
import tifffile

from napari_tissuekit.core import segmentation


class FakeModel:
    def __init__(self, *a, **k):
        pass

    def eval(self, img, **k):
        m = np.zeros(img.shape, np.uint16)
        m[2:8, 2:8] = 1
        return (m,)


@pytest.fixture
def movie(tmp_path, monkeypatch):
    monkeypatch.setattr(segmentation.models, "CellposeModel", FakeModel)
    monkeypatch.setattr(segmentation, "cell_outline_stack", lambda masks, progress=None: np.zeros(masks.shape, np.uint8))
    path = tmp_path / "m.tif"
    tifffile.imwrite(path, np.zeros((10, 16, 16), np.uint8))
    return path


def _count_replaces(monkeypatch):
    calls = []
    real = os.replace
    monkeypatch.setattr(segmentation.os, "replace", lambda a, b: (calls.append(b), real(a, b))[1])
    return calls


def test_checkpoint_every_n_frames_not_every_frame(movie, tmp_path, monkeypatch):
    monkeypatch.setenv("TISSUEKIT_CHECKPOINT_EVERY", "4")
    calls = _count_replaces(monkeypatch)
    segmentation.main(["--input", str(movie), "--output-dir", str(tmp_path / "o"), "--force-cpu"])
    assert len(calls) == 2  # after frame 4 and 8, not 10 times
    assert (tmp_path / "o" / "m_cp_masks.tif").exists()
    assert not list((tmp_path / "o").glob("*partial*")) and not (tmp_path / "o" / "progress.txt").exists()


def test_sigterm_saves_checkpoint_and_run_resumes(movie, tmp_path, monkeypatch):
    out = tmp_path / "o"
    monkeypatch.setenv("TISSUEKIT_CHECKPOINT_EVERY", "100")  # never reached: only SIGTERM saves
    seen = {"n": 0}
    real_eval = FakeModel.eval

    def eval_then_term(self, img, **k):
        seen["n"] += 1
        if seen["n"] == 7:
            signal.raise_signal(signal.SIGTERM)  # os.kill on Windows = TerminateProcess: kills pytest, no handler
        return real_eval(self, img, **k)

    monkeypatch.setattr(FakeModel, "eval", eval_then_term)
    with pytest.raises(SystemExit) as e:
        segmentation.main(["--input", str(movie), "--output-dir", str(out), "--force-cpu"])
    assert e.value.code == 128 + signal.SIGTERM
    assert (out / "progress.txt").read_text() == "6"  # frames 1-6 were finished
    assert np.load(out / "m_cp_masks_partial.npy").shape == (10, 16, 16)
    monkeypatch.setattr(FakeModel, "eval", real_eval)
    segmentation.main(["--input", str(movie), "--output-dir", str(out), "--force-cpu"])  # resumes at frame 6
    assert tifffile.imread(out / "m_cp_masks.tif").shape == (10, 16, 16)
