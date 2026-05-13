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


@dataclass(frozen=True)
class Chromagram:
    """A chromagram with the metadata needed downstream."""

    X: np.ndarray            # (12, num_frames)
    feature_rate: float      # frames per second
    duration: float          # seconds
    sr: int
    hop_length: int
    variant: str


def normalize_columns(X: np.ndarray, norm: str | None = "2", eps: float = 1e-10) -> np.ndarray:
    """L1, L2, or max-normalize each column. Zero-norm columns are left as zero."""
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
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 2048,
    gamma: float | None = None,
    norm: str | None = "2",
) -> Chromagram:
    """Load an audio file and compute its chromagram.

    Parameters
    ----------
    audio_path : str
        Path to a wav/mp3/flac file.
    variant : {"STFT", "IIR", "CQT"}
        Feature variant. Default "CQT" usually gives cleaner piano chroma;
        "STFT" matches the FMP notebook's first example.
    sr : int
        Target sample rate.
    n_fft, hop_length : int
        Frame and hop length. Defaults match FMP §5.2.
    gamma : float, optional
        Log compression: log(1 + gamma * |S|^2) for STFT; log(1 + gamma * S) for IIR.
        Ignored for CQT.
    norm : {"1", "2", "max", None}
        Per-frame normalization of the final chromagram.
    """
    import librosa  # lazy: slow to import, not needed for the rest of the package

    x, sr = librosa.load(audio_path, sr=sr)
    duration = x.shape[0] / sr

    if variant == "STFT":
        S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, pad_mode="constant", center=True)
        S = np.abs(S) ** 2
        if gamma is not None:
            S = np.log(1 + gamma * S)
        X = librosa.feature.chroma_stft(
            S=S, sr=sr, tuning=0, norm=None, hop_length=hop_length, n_fft=n_fft
        )
    elif variant == "CQT":
        X = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length, norm=None)
    elif variant == "IIR":
        S = librosa.iirt(y=x, sr=sr, win_length=n_fft, hop_length=hop_length, center=True, tuning=0.0)
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
        feature_rate=sr / hop_length,
        duration=duration,
        sr=sr,
        hop_length=hop_length,
        variant=variant,
    )
