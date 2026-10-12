"""Cellpose + TA-logic cell tracker.

Uses Tissue Analyzer's pyramidal motion prior and neighbour-topology score,
but replaces greedy point-probe matching with global LAP assignment over an
overlap matrix. No epyseg, no TA folder layout, no RGB24 identity.
"""

from __future__ import annotations
