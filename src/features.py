"""Chroma feature extraction.

Three variants, all returning an (12, num_frames) chromagram L2-normalized
per frame (the standard FMP §5.2 setup):

* STFT — short-time Fourier transform with optional log compression.
* IIR  — elliptic IIR filter-bank decomposition (librosa.iirt).
* CQT  — constant-Q transform.

Hop and frame defaults follow FMP (N=4096, H=2048 at 22050 Hz, ~10.77 Hz rate).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Variant = Literal["STFT", "IIR", "CQT"]


@dataclass
class Chromagram:
    """A chromagram with the metadata needed downstream.

    :param X: Chroma matrix of shape (num_chroma, num_frames).
    :param feature_rate: Frames per second.
    :param duration: Audio duration in seconds.
    :param sample_rate: Audio sample rate in Hz.
    :param hop_length: STFT hop length in samples.
    :param variant: Which chroma variant produced this object ("STFT", "IIR", or "CQT").
    """

    X: np.ndarray
    feature_rate: float
    duration: float
    sample_rate: int
    hop_length: int
    variant: str


def normalize_columns(X: np.ndarray, norm: str | None = "2", eps: float = 1e-10) -> np.ndarray:
    """L1, L2, or max-normalize each column. Zero-norm columns are left as zero.

    :param X: Matrix whose columns will be normalized.
    :param norm: Norm to apply per column: ``"1"`` (sum of absolute values), ``"2"``
        (Euclidean), ``"max"`` (max of absolute values), or ``None`` to skip
        normalization and return ``X`` unchanged.
    :param eps: Columns with norm below ``eps`` are treated as zero and left untouched.
    """
    if norm is None:
        return X
    if norm == "1":
        s = np.sum(np.abs(X), axis=0, keepdims=True)
    elif norm == "2":
        s = np.linalg.norm(X, axis=0, keepdims=True)
    elif norm == "max":
        s = np.max(np.abs(X), axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown norm {norm!r}; use '1', '2', 'max', or None.")
    s = np.where(s < eps, 1.0, s)
    return X / s


def compute_chromagram(
    audio_path: str,
    *,
    variant: Variant = "CQT",
    sample_rate: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 2048,
    gamma: float | None = None,
    norm: str | None = "2",
) -> Chromagram:
    """Load an audio file and compute its chromagram.

    :param audio_path: Path to a wav/mp3/flac file.
    :param variant: Feature variant — one of ``"STFT"``, ``"IIR"``, or ``"CQT"``.
        Default ``"CQT"`` usually gives cleaner piano chroma; ``"STFT"`` matches
        the FMP notebook's first example.
    :param sample_rate: Target sample rate in Hz; audio is resampled on load.
    :param n_fft: STFT window length in samples (also the FFT size).
    :param hop_length: STFT hop length in samples. Defaults match FMP §5.2.
    :param gamma: Log-compression factor: ``log(1 + gamma * |S|^2)`` for STFT,
        ``log(1 + gamma * S)`` for IIR. Ignored for CQT. ``None`` disables it.
    :param norm: Per-frame normalization of the final chromagram. One of
        ``"1"``, ``"2"``, ``"max"``, or ``None``.
    """
    import librosa  # lazy: slow to import, not needed for the rest of the package

    x, sample_rate = librosa.load(audio_path, sr=sample_rate)
    duration = x.shape[0] / sample_rate

    if variant == "STFT":
        S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, pad_mode="constant", center=True)
        S = np.abs(S) ** 2
        if gamma is not None:
            S = np.log(1 + gamma * S)
        X = librosa.feature.chroma_stft(
            S=S, sr=sample_rate, tuning=0, norm=None, hop_length=hop_length, n_fft=n_fft
        )
    elif variant == "CQT":
        X = librosa.feature.chroma_cqt(y=x, sr=sample_rate, hop_length=hop_length, norm=None)
    elif variant == "IIR":
        S = librosa.iirt(y=x, sr=sample_rate, win_length=n_fft, hop_length=hop_length, center=True, tuning=0.0)
        if gamma is not None:
            S = np.log(1.0 + gamma * S)
        X = librosa.feature.chroma_cqt(
            C=S, bins_per_octave=12, n_octaves=7, fmin=librosa.midi_to_hz(24), norm=None
        )
    else:
        raise ValueError(f"Unknown variant {variant!r}; expected STFT, IIR, or CQT.")

    X = normalize_columns(X, norm=norm)

    return Chromagram(
        X=X,
        feature_rate=sample_rate / hop_length,
        duration=duration,
        sample_rate=sample_rate,
        hop_length=hop_length,
        variant=variant,
    )
