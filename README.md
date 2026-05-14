# fmp-chord-recognition

Chord recognition for classical solo piano, planned to have three stages:

1. [In-progress] **Template-based baseline** (FMP §5.2): chroma features → cosine similarity against 24 binary major/minor triad templates → argmax.
2. [To-do] **HMM postfilter** (FMP §5.3)
3. [To-do] **Neural model** (PyTorch)

**Validation target.** Chopin, Étude Op. 25 No. 1 in A♭ major ("Aeolian Harp") (annotation in-progress).

Status: Evaluation is in progress — beat-tracked aggregation and timestamp-aligned annotations are the planned approach
to properly evaluate chords from arpeggios in solo piano. More comments on this in notebooks/02_template_recognition.ipynb.

## Layout

```
fmp-chord-recognition/
├── src/                   # importable modules
│   ├── features.py        # STFT / IIR / CQT chromagrams
│   ├── templates.py       # 24 binary triad templates + labels
│   ├── recognition.py     # template-based recognition
│   ├── annotations.py     # libfmp CSV / MIREX .lab I/O
│   └── visualization.py   # plotting
├── notebooks/             # exploratory work
└── data/                  # gitignored
    ├── audio/             # mp3s here
    ├── annotations/       # annotations from various sources
    └── predictions/       # predictions from models

To be added: CLI & tests
```

## References

- Müller, *Fundamentals of Music Processing*, Springer 2015 — Chapter 5.
- to read: [McLeod & Rohrmeier (2024) Detecting chord tone alterations and suspensions](https://www.tandfonline.com/doi/full/10.1080/09298215.2024.2412595#d1e156)
