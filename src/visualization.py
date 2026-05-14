"""Plotting utilities for chromagrams, similarity matrices, and annotations.

Pure matplotlib — no runtime dependency on libfmp.
"""

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.annotations import Interval
from src.templates import chroma_names, get_chord_labels


def plot_chromagram(
    X: np.ndarray,
    feature_rate: float,
    *,
    ax: Axes | None = None,
    title: str = "",
    cmap: str = "gray_r",
    clim: tuple[float, float] = (0.0, 1.0),
    use_flats: bool = False,
    reference_root: int = 0,
    vocab: Iterable[str] | None = None,
) -> Axes:
    """Plot a (12, N) chromagram."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.5))

    labels = get_chord_labels(use_flats=use_flats, reference_root=reference_root, vocab=vocab)
    n_frames = X.shape[1]
    extent = (0, n_frames / feature_rate, -0.5, 11.5)
    ax.imshow(X, aspect="auto", origin="lower", cmap=cmap, extent=extent, clim=clim)
    ax.set_yticks(range(12))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Chroma")
    if title:
        ax.set_title(title)
    return ax


def plot_chord_similarity(
    chord_sim: np.ndarray,
    feature_rate: float,
    *,
    nonchord: bool = False,
    ax: Axes | None = None,
    title: str = "",
    cmap: str = "gray_r",
    use_flats: bool = False,
    reference_root: int = 0,
) -> Axes:
    """Plot a (num_chords, N) time–chord similarity matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    labels = get_chord_labels(nonchord=nonchord, use_flats=use_flats, reference_root=reference_root)
    n_frames = chord_sim.shape[1]
    extent = (0, n_frames / feature_rate, -0.5, len(labels) - 0.5)
    ax.imshow(chord_sim, aspect="auto", origin="lower", cmap=cmap, extent=extent)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Chord")
    if title:
        ax.set_title(title)
    return ax


def overlay_annotations(
    ax: Axes,
    annotations: list[Interval],
    *,
    color_map: Mapping[str, str] | None = None,
    alpha: float = 0.25,
    show_labels: bool = False,
) -> None:
    """Overlay coloured rectangles for chord segments on an existing axes."""
    ymin, ymax = ax.get_ylim()
    default_color = "tab:gray"
    for start, end, label in annotations:
        color = color_map.get(label, default_color) if color_map else default_color
        ax.axvspan(start, end, ymin=0, ymax=1, color=color, alpha=alpha, linewidth=0)
        if show_labels:
            ax.text(
                (start + end) / 2,
                ymax - (ymax - ymin) * 0.05,
                label,
                ha="center",
                va="top",
                fontsize=8,
            )


def plot_recognition_summary(
    chromagram: np.ndarray,
    chord_sim: np.ndarray,
    feature_rate: float,
    *,
    reference: list[Interval] | None = None,
    color_map: Mapping[str, str] | None = None,
    nonchord: bool = False,
    use_flats: bool = False,
    reference_root: int = 0,
) -> Figure:
    """Three-panel summary: chromagram → chord similarity → reference strip."""
    n_panels = 3 if reference is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 7), constrained_layout=True)

    plot_chromagram(chromagram, feature_rate, ax=axes[0], title="Chromagram", use_flats=use_flats, reference_root=reference_root)
    if reference is not None:
        overlay_annotations(axes[0], reference, color_map=color_map, alpha=0.15)

    plot_chord_similarity(
        chord_sim,
        feature_rate,
        ax=axes[1],
        nonchord=nonchord,
        title="Chord similarity",
        use_flats=use_flats,
        reference_root=reference_root,
    )
    if reference is not None:
        overlay_annotations(axes[1], reference, color_map=color_map, alpha=0.15)

        ax_ann = axes[2]
        ax_ann.set_yticks([])
        ax_ann.set_xlim(axes[1].get_xlim())
        ax_ann.set_xlabel("Time (s)")
        ax_ann.set_title("Reference annotations")
        overlay_annotations(ax_ann, reference, color_map=color_map, alpha=0.6, show_labels=True)

    return fig
