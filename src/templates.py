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


def chroma_names(*, use_flats: bool = False) -> tuple[str, ...]:
    """Return the 12-element pitch-class name tuple in either sharp or flat spelling.

    The natural notes (C, D, E, F, G, A, B) are unchanged; only the five
    chromatic pitch classes differ. Sharps are the default (matching FMP §5.2);
    flats are typically more readable in flat-key repertoire (e.g. Ab major).
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
    "dim7":  (0, 3, 6, 9),      # fully diminished 7th  (vii°7)
    "hdim7": (0, 3, 6, 10),     # half-diminished 7th   (ø7, a.k.a. m7♭5)
    "mmaj7": (0, 3, 7, 11),     # minor-major 7th       (Cm(maj7): used in harmonic-minor contexts, Chopin/Wagner)
    "aug7":  (0, 4, 8, 10),     # augmented 7th         (a.k.a. V+7; Schumann/Wagner, not jazz-only)
    "fr6":   (0, 4, 6, 10),     # French augmented 6th  (chord-tone set {0,4,6,10}; see note below)
    "sus4":  (0, 5, 7),         # suspended 4th
    "sus2":  (0, 2, 7),         # suspended 2nd
}

# Preset vocabularies. Order is preserved in the label list and template columns.
VOCAB_TRIADS: list[str] = ["", "m"]                                                # 24 chords (FMP §5.2 default)
VOCAB_TRIADS_SEVENTHS: list[str] = ["", "m", "7", "maj7", "m7"]                    # 60 chords
VOCAB_CLASSICAL: list[str] = [                                                     # 144 chords
    "", "m", "dim", "aug",                # all four triad qualities
    "7", "maj7", "m7", "dim7", "hdim7",   # tonal 7th chords
    "mmaj7", "aug7",                      # harmonic-minor / chromatic-romantic colour
    "sus4",                               # 4-3 suspensions (the dominant suspension type)
]
# Why sus4 but not sus2:
# In chroma space, sus2 and sus4 are the same template at different roots
# (Csus2 = C-D-G = chroma {0,2,7} = Gsus4 = G-C-D = chroma {0,2,7}). Including
# both creates redundant columns that split probability mass arbitrarily. Sus4
# wins by convention (chords are named by root/bass, 4-3 suspensions dominate
# in the period). 2-3 suspensions are rare enough that the occasional
# mis-rooted Csus2 → Gsus4 label is an acceptable cost.
VOCAB_EXTENDED: list[str] = [                                                      # 168 chords; adds fr6 + sus2
    *VOCAB_CLASSICAL, "fr6", "sus2",
]


def _quality_to_binary(quality: str, root: str) -> np.ndarray:
    """
    Build the binary chroma vector for a single quality with the provided root.

    root: One of the 12 chromatic pitch classes, e.g. "C", "F#", "Bb".
    """
    root_mapping = {name: i for i, name in enumerate(chroma_names())}
    if root not in root_mapping:
        raise ValueError(f"Unknown root '{root}'. Must be one of {chroma_names()}.")
    
    # TODO: incorporate root into the template generation logic instead of rolling later

    v = np.zeros(12, dtype=np.float64)
    for interval in QUALITY_INTERVALS[quality]:
        v[interval % 12] = 1.0
    return v


# Reference templates rooted at C, kept for explicit access (and for tests).
TEMPLATE_C_MAJ: np.ndarray = _quality_to_binary("")
TEMPLATE_C_MIN: np.ndarray = _quality_to_binary("m")

# template in A flat major: apply a cyclic shift of 8 semitones to the C-rooted template
TEMPLATE_AB_MAJ: np.ndarray = np.roll(TEMPLATE_C_MAJ, 8)


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
    vocab: Iterable[str] = VOCAB_CLASSICAL,  # TODO: make vocab a required argument to avoid accidental mismatches between labels and templates
    *,
    nonchord: bool = False,
    use_flats: bool = False,
) -> list[str]:
    """Return the ordered list of chord labels for the given vocabulary.

    Order: for each quality (in vocabulary order), all 12 roots. So for the
    default triad vocabulary you get [C, C#, ..., B, Cm, C#m, ..., Bm]; for
    `VOCAB_TRIADS_SEVENTHS` you continue with [C7, C#7, ..., Cmaj7, ...].

    With `use_flats=True`, chromatic root names switch to their flat spelling
    (Db, Eb, Gb, Ab, Bb) — typically preferred in flat-key repertoire.
    """
    vocab = _validate_vocab(vocab)
    names = chroma_names(use_flats=use_flats)
    labels: list[str] = []
    for quality in vocab:
        labels.extend(f"{root}{quality}" for root in names)
    if nonchord:
        labels.append("N")
    return labels


def generate_chord_templates(
    vocab: Iterable[str] = VOCAB_CLASSICAL,
    *,
    nonchord: bool = False,
) -> np.ndarray:
    """Build the chord-template matrix of shape (12, 12 * len(vocab) [+ 1]).

    Each column is the binary chroma vector for one chord, in the order given by
    :func:`get_chord_labels`. The optional non-chord template is all-ones.

    Note on cardinality bias: chord templates with more notes (e.g. 7ths) have
    larger Euclidean norms before normalization. L2-normalizing partially fixes
    this, but in practice cosine similarity still tilts toward larger-cardinality
    chords whenever the chroma signal contains overtones or pedal blur. See
    Cho & Bello (2014) for a treatment, and consider non-binary templates or
    HMM postfiltering before relying on the extended vocabulary.
    """
    vocab = _validate_vocab(vocab)
    num_qualities = len(vocab)
    num_chords = 12 * num_qualities + (1 if nonchord else 0)
    templates = np.zeros((12, num_chords), dtype=np.float64)

    for q_idx, quality in enumerate(vocab):
        base = _quality_to_binary(quality)
        for root in range(12):
            templates[:, q_idx * 12 + root] = np.roll(base, root)

    if nonchord:
        templates[:, -1] = 1.0  # all-ones catch-all

    return templates
