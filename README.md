# fmp-chord-recognition

Chord recognition for classical solo piano, built up in three stages:

1. **Template-based baseline** (FMP §5.2): chroma features → cosine similarity against 24 binary major/minor triad templates → argmax.
2. **HMM postfilter** (FMP §5.3): to-do
3. **Neural model** (PyTorch): to-do

**Validation target.** Chopin, Étude Op. 25 No. 1 in A♭ major ("Aeolian Harp").
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
├── scripts/               # CLI entry points
├── tests/
└── data/
    ├── audio/             # gitignored — drop mp3s here
    ├── annotations/       # committed — your ground-truth labels
    └── predictions/       # gitignored
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Importing from notebooks

From a notebook in `notebooks/`, two lines at the top:

```python
import sys
sys.path.insert(0, "..")

from chordrec.features import compute_chromagram
from chordrec.visualization import plot_chromagram
```

That adds the project root to `sys.path`, so `chordrec` (the directory) becomes importable as a module. No package install needed.

## CLI

```bash
python scripts/predict.py --audio data/audio/op25_no1.mp3 --variant CQT \
    --out data/predictions/op25_no1.csv

python scripts/evaluate.py --ref data/annotations/op25_no1.csv \
    --est data/predictions/op25_no1.csv
```

## Tests

```bash
python -m pytest tests/
```

## Annotation format

CSV (libfmp-compatible) or `.lab` (MIREX/Isophonics). See `data/annotations/README.md`.

## References

- Müller, *Fundamentals of Music Processing*, Springer 2015 — Chapter 5.
- Cho & Bello, "On the Relative Importance of Individual Components of Chord Recognition Systems," *IEEE/ACM TASLP* 22 (2014).
- Jiang et al., "Analyzing Chroma Feature Types for Automated Chord Recognition," AES Semantic Audio 2011.
