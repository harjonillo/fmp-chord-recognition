"""Chord vocabulary and binary templates.

Each chord quality is defined as a set of semitone intervals from the root.
A *vocabulary* is an ordered list of quality keys; the full chord set is the
cartesian product of vocabulary × 12 roots, optionally with a non-chord catch-all.

To add a new chord type:

1. Add an entry to `QUALITY_INTERVALS` with its semitone offsets from the root.
2. (Optionally) add it to a vocabulary list.

Label format: `<root><quality_suffix>`. The major quality has an empty suffix
(`C`, `F#`), so `C` means C major and `Cm` means C minor, matching FMP §5.2.
"""

from typing import Iterable

import numpy as np

CHROMA_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
CHROMA_NAMES_FLAT: tuple[str, ...] = (
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
)

# Reference roots — the semitone offset (0-indexed from C) used as the anchor
# when generating templates. Default is C (FMP convention); A♭ matches the
# tonic of Op. 25 No. 1 and makes the first template column be the tonic chord.
REFERENCE_ROOT_C: int = 0
REFERENCE_ROOT_AB: int = 8


def chroma_names(*, use_flats: bool = True) -> tuple[str, ...]:
    """Return the 12-element pitch-class name tuple in either sharp or flat spelling.

    The natural notes (C, D, E, F, G, A, B) are unchanged; only the five
    chromatic pitch classes differ. Flat spelling is the package default
    (set for flat-key classical repertoire); pass ``use_flats=False`` for
    the FMP §5.2 sharp convention.

    :param use_flats: If ``True`` (default), return ``("C", "Db", "D", "Eb", ...)``.
        If ``False``, return the sharp variant ``("C", "C#", "D", "D#", ...)``.
    """
    return CHROMA_NAMES_FLAT if use_flats else CHROMA_NAMES

# Semitone offsets from the root, indexed by quality suffix.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "":      (0, 4, 7),         # major triad
    "m":     (0, 3, 7),         # minor triad
    "7":     (0, 4, 7, 10),     # dominant 7th
    "maj7":  (0, 4, 7, 11),     # major 7th
    "m7":    (0, 3, 7, 10),     # minor 7th
    "dim":   (0, 3, 6),         # diminished triad
    "aug":   (0, 4, 8),         # augmented triad
    "dim7":  (0, 3, 6, 9),      # fully diminished 7th
    "hdim7": (0, 3, 6, 10),     # half-diminished 7th   
    "mmaj7": (0, 3, 7, 11),     # minor-major 7th       
    "aug7":  (0, 4, 8, 10),     # augmented 7th         
    "fr6":   (0, 4, 6, 10),     # French augmented 6th  
    "sus4":  (0, 5, 7),         # suspended 4th
    "7sus4":  (0, 5, 10),       # dominant 7th, suspended 4th
    "sus2":  (0, 2, 7),         # suspended 2nd
}

# Preset vocabularies. Order is preserved in the label list and template columns.
VOCAB_CHROMA: list[str] = [""]                                                     # 12 chords (chroma templates only)
VOCAB_TRIADS: list[str] = ["", "m"]                                                # 24 chords (FMP §5.2 default)
VOCAB_TRIADS_SEVENTHS: list[str] = ["", "m", "7", "maj7", "m7"]                    
VOCAB_CLASSICAL: list[str] = [                                                     
    "", "m", "dim", "aug",                
    "7", "maj7", "m7", "dim7", "hdim7", "7sus4", "sus4",  
    "aug7", #"mmaj7"                
]
VOCAB_EXTENDED: list[str] = [                                                      
    *VOCAB_CLASSICAL, "fr6", "sus2",
]


def _quality_to_binary(quality: str) -> np.ndarray:
    """Build the binary chroma vector for a single quality with root C."""
    v = np.zeros(12, dtype=np.float64)
    for interval in QUALITY_INTERVALS[quality]:
        v[interval % 12] = 1.0
    return v


# Reference templates rooted at C, kept for explicit access (and for tests).
TEMPLATE_C_MAJ: np.ndarray = _quality_to_binary("")
TEMPLATE_C_MIN: np.ndarray = _quality_to_binary("m")


def _validate_vocab(vocab: Iterable[str]) -> list[str]:
    """Resolve a vocabulary list and check every entry is a known quality."""
    vocab = list(vocab)
    unknown = [q for q in vocab if q not in QUALITY_INTERVALS]
    if unknown:
        raise ValueError(
            f"Unknown chord qualities: {unknown}. "
            f"Available: {sorted(QUALITY_INTERVALS)}"
        )
    return vocab


def get_chord_labels(
    vocab: Iterable[str] | None = None,
    *,
    nonchord: bool = False,
    use_flats: bool = True,
    reference_root: int = REFERENCE_ROOT_C,
) -> list[str]:
    """Return the ordered list of chord labels for the given vocabulary.

    Order: for each quality (in vocabulary order), all 12 roots starting from
    ``reference_root`` and ascending by semitone. With the default
    ``reference_root=0`` (C), the major block is ``[C, Db, D, ..., B]``. With
    ``reference_root=8`` (A♭), the major block is
    ``[Ab, A, Bb, B, C, Db, D, Eb, E, F, Gb, G]`` — i.e. the tonic chord of
    A♭ major sits at column 0.

    :param vocab: Iterable of quality suffixes (e.g. ``["", "m", "7"]``). Each
        must be a key of :data:`QUALITY_INTERVALS`. Defaults to
        :data:`VOCAB_CLASSICAL` (144 chords).
    :param nonchord: If ``True``, append the non-chord label ``"N"`` at the end.
    :param use_flats: If ``True`` (default), use flat spelling for roots
        (Db, Eb, Gb, Ab, Bb); otherwise sharp spelling.
    :param reference_root: Semitone offset (0-11) of the first root in each
        quality block.
    """
    if vocab is None:
        vocab = VOCAB_CLASSICAL
    vocab = _validate_vocab(vocab)
    names = chroma_names(use_flats=use_flats)
    rotated_names = tuple(names[(reference_root + i) % 12] for i in range(12))
    labels: list[str] = []
    for quality in vocab:
        labels.extend(f"{root}{quality}" for root in rotated_names)
    if nonchord:
        labels.append("N")
    return labels


def generate_chord_templates(
    vocab: Iterable[str] | None = None,
    *,
    nonchord: bool = False,
    reference_root: int = REFERENCE_ROOT_C,
) -> np.ndarray:
    """Build the chord-template matrix of shape ``(12, 12 * len(vocab) [+ 1])``.

    Column ordering: for each quality, the 12 roots cycle starting from
    ``reference_root`` (in semitones from C). With ``reference_root=8``, column
    0 is the A♭-rooted chord of the first quality — i.e. A♭ major when the
    first quality is ``""``. The template *contents* (which chroma bins each
    column activates) are unchanged: only the column ordering shifts.

    Defaults to :data:`VOCAB_CLASSICAL` (144 chords). For the FMP §5.2 baseline
    pass ``vocab=VOCAB_TRIADS`` explicitly.

    Each column is the binary chroma vector for one chord, in the order given by
    :func:`get_chord_labels`. The optional non-chord template is all-ones.

    Note on cardinality bias: chord templates with more notes (e.g. 7ths) have
    larger Euclidean norms before normalization. L2-normalizing partially fixes
    this, but in practice cosine similarity still tilts toward larger-cardinality
    chords whenever the chroma signal contains overtones or pedal blur. See
    Cho & Bello (2014) for a treatment, and consider non-binary templates or
    HMM postfiltering before relying on the extended vocabulary.

    :param vocab: Iterable of quality suffixes. Each must be a key of
        :data:`QUALITY_INTERVALS`. Defaults to :data:`VOCAB_CLASSICAL`.
    :param nonchord: If ``True``, append an all-ones non-chord template as the
        final column.
    :param reference_root: Semitone offset (0-11) used as the starting root of
        each quality block (see "Column ordering" above).
    """
    if vocab is None:
        vocab = VOCAB_CLASSICAL
    vocab = _validate_vocab(vocab)
    num_qualities = len(vocab)
    num_chords = 12 * num_qualities + (1 if nonchord else 0)
    templates = np.zeros((12, num_chords), dtype=np.float64)

    for q_idx, quality in enumerate(vocab):
        base = _quality_to_binary(quality)
        for offset in range(12):
            root = (reference_root + offset) % 12
            templates[:, q_idx * 12 + offset] = np.roll(base, root)

    if nonchord:
        templates[:, -1] = 1.0  # all-ones catch-all

    return templates