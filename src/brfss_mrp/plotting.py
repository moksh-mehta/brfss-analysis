"""Shared matplotlib styling for report figures (static PNGs, light surface)."""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
        "font.size": 9, "text.color": TEXT,
    })
