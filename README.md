# fmp-chord-recognition

Chord recognition for classical solo piano, built up in three stages:

1. [In-progress] **Template-based baseline** (FMP §5.2): chroma features → cosine similarity against 24 binary major/minor triad templates → argmax.
2. [To-do] **HMM postfilter** (FMP §5.3)
3. [To-do] **Neural model** (PyTorch)

**Validation target.** Chopin, Étude Op. 25 No. 1 in A♭ major ("Aeolian Harp") (annotation in-progress).

## Layout

```
fmp-chord-recognition/
├── src/              # importable modules
│   ├── features.py        # STFT / IIR / CQT chromagrams
│   ├── templates.py       # 24 binary triad templates + labels
│   ├── recognition.py       # template-based recognition
│   ├── annotations.py     # libfmp CSV / MIREX .lab I/O
│   ├── evaluation.py        # mir_eval wrappers + confusion
│   └── visualization.py   # plotting
├── notebooks/             # exploratory work
└── data/
    ├── audio/             # gitignored — drop mp3s here
    ├── annotations/       # committed — your ground-truth labels
    └── predictions/       # gitignored

To be added: CLI & tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Importing from notebooks

From a notebook in `notebooks/`, two lines at the top:

```python
import sys
sys.path.insert(0, "..")

from src.features import compute_chromagram
from src.visualization import plot_chromagram
```

## Annotation format

CSV (libfmp-compatible) or `.lab` (MIREX/Isophonics). See `data/annotations/README.md`.

## References

- Müller, *Fundamentals of Music Processing*, Springer 2015 — Chapter 5.
- to read: [McLeod & Rohrmeier (2024) Detecting chord tone alterations and suspensions](https://www.tandfonline.com/doi/full/10.1080/09298215.2024.2412595#d1e156)
