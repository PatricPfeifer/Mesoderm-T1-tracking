"""Napari interface and command line for Cellpose segmentation, tracking and T1 detection.

``napari_tissuekit.core`` and ``napari_tissuekit.cli`` need neither napari nor Qt, so they
run on a headless HPC node. The widget is only imported when it is asked for.
"""

__version__ = "0.1.0"
__all__ = ["TissueKitWidget"]


def __getattr__(name):
    if name == "TissueKitWidget":
        from ._widget import TissueKitWidget

        return TissueKitWidget
    raise AttributeError(name)
