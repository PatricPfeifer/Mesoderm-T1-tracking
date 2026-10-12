"""Napari interface and command line for Cellpose segmentation, tracking and T1 detection.

``napari_patric.core`` and ``napari_patric.cli`` need neither napari nor Qt, so they
run on a headless HPC node. The widget is only imported when it is asked for.
"""

__version__ = "0.1.0"
__all__ = ["PatricWidget"]


def __getattr__(name):
    if name == "PatricWidget":
        from ._widget import PatricWidget

        return PatricWidget
    raise AttributeError(name)
