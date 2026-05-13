"""Template-based chord recognition.

Mirrors FMP §5.2: cosine similarity between each chroma frame and each chord
template, then argmax per frame. The HMM postfilter from §5.3 will plug in by
taking `chord_sim` as observation likelihoods instead of applying the argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np

from src.features import Chromagram, normalize_columns
from src.templates import VOCAB_TRIADS, generate_chord_templates, get_chord_labels


@dataclass(frozen=True)
class RecognitionResult:
    """Frame-wise chord recognition output."""

    chord_sim: np.ndarray          # (num_chords, num_frames)
    chord_max: np.ndarray          # (num_chords, num_frames) — only the max sim per frame is nonzero
    chord_progression: list[int]   # chord indices with consecutive duplicates removed
    chord_indices: np.ndarray      # (num_frames,) — argmax chord index per frame
    labels: list[str]              # vocabulary, indexed by chord_indices
    feature_rate: float            # frames per second

    @property
    def predicted_labels(self) -> list[str]:
        return [self.labels[i] for i in self.chord_indices]

    def to_intervals(self) -> list[tuple[float, float, str]]:
        """Collapse frames to (start, end, label) intervals in seconds.

        Consecutive frames with the same label are merged.
        """
        intervals: list[tuple[float, float, str]] = []
        frame_dur = 1.0 / self.feature_rate
        frame_idx = 0
        for label_idx, group in groupby(self.chord_indices):
            n = sum(1 for _ in group)
            start = frame_idx * frame_dur
            end = (frame_idx + n) * frame_dur
            intervals.append((start, end, self.labels[label_idx]))
            frame_idx += n
        return intervals


def recognize_template(
    chroma: Chromagram | np.ndarray,
    *,
    feature_rate: float | None = None,
    vocab: list[str] | None = None,
    norm_sim: str | None = "1",
    nonchord: bool = False,
    use_flats: bool = False,
) -> RecognitionResult:
    """Recognize chords by template matching.

    Parameters
    ----------
    chroma : Chromagram or np.ndarray
        A Chromagram from `compute_chromagram`, or a raw (12, num_frames) array
        (in which case `feature_rate` must be provided).
    feature_rate : float, optional
        Required when `chroma` is a raw array.
    vocab : list[str], optional
        Chord qualities to recognize. Defaults to `VOCAB_TRIADS` (maj/min, 24
        chords). See `chordrec.templates` for predefined extended vocabularies.
    norm_sim : {"1", "2", "max", None}
        Normalization of the per-frame similarity vector across chords. FMP uses
        "1" (similarities sum to 1 per frame, pseudo-posterior) or "max" (best
        chord = 1).
    nonchord : bool
        If True, include the all-ones non-chord template (label "N").
    use_flats : bool
        If True, use flat spelling for chromatic roots (Db, Eb, Gb, Ab, Bb)
        instead of sharps. The templates themselves are unchanged; only the
        output labels differ.
    """
    if vocab is None:
        vocab = VOCAB_TRIADS

    if isinstance(chroma, Chromagram):
        X = chroma.X
        fr = chroma.feature_rate
    else:
        if feature_rate is None:
            raise ValueError("feature_rate must be provided when chroma is a raw array.")
        X = chroma
        fr = feature_rate

    if X.shape[0] != 12:
        raise ValueError(f"Expected chromagram with 12 rows, got shape {X.shape}.")

    templates = generate_chord_templates(vocab, nonchord=nonchord)
    X_norm = normalize_columns(X, norm="2")
    T_norm = normalize_columns(templates, norm="2")
    chord_sim = T_norm.T @ X_norm  # (num_chords, num_frames)

    if norm_sim is not None:
        chord_sim = normalize_columns(chord_sim, norm=norm_sim)

    chord_indices = np.argmax(chord_sim, axis=0).astype(np.int32)
    labels = get_chord_labels(vocab, nonchord=nonchord, use_flats=use_flats)

    # add predicted chords based on the max similarity
    chord_max = np.zeros_like(chord_sim)
    for i, idx in enumerate(chord_indices):
        chord_max[idx, i] = chord_sim[idx, i]
    
    # add chord progression, removing consecutive duplicates
    chord_progression = []
    for i, idx in enumerate(chord_indices):
        if not chord_progression or chord_progression[-1] != idx:
            chord_progression.append(idx)

    return RecognitionResult(
        chord_sim=chord_sim,
        chord_max=chord_max,
        chord_progression=chord_progression,
        chord_indices=chord_indices,
        labels=labels,
        feature_rate=fr,
    )
