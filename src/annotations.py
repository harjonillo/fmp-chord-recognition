"""Read and write chord annotations.

Two formats:

* csv (libfmp-style): comma-separated with header `start,end,label`.
* lab (MIREX/Isophonics): tab-separated, no header.

Format is detected by file extension; pass `fmt` to override.
"""

from pathlib import Path

import pandas as pd

Interval = tuple[float, float, str]


def read_annotations(path: str | Path, *, fmt: str | None = None) -> list[Interval]:
    """Read a chord annotation file into a list of ``(start, end, label)`` tuples.

    :param path: Path to the annotation file.
    :param fmt: Either ``"csv"`` or ``"lab"``. If ``None`` (default), inferred
        from the file extension.
    """
    path = Path(path)
    fmt = fmt or _infer_format(path)

    if fmt == "csv":
        df = pd.read_csv(path)
        required = {"start", "end", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
    elif fmt == "lab":
        df = pd.read_csv(path, sep="\t", header=None, names=["start", "end", "label"])
    else:
        raise ValueError(f"Unknown annotation format {fmt!r}; expected 'csv' or 'lab'.")

    return [(float(r.start), float(r.end), str(r.label)) for r in df.itertuples(index=False)]


def write_annotations(
    intervals: list[Interval],
    path: str | Path,
    *,
    fmt: str | None = None,
) -> None:
    """Write intervals to disk in the chosen format.

    :param intervals: List of ``(start, end, label)`` tuples to write.
    :param path: Destination path. Parent directories are created if needed.
    :param fmt: Either ``"csv"`` or ``"lab"``. If ``None`` (default), inferred
        from the file extension.
    """
    path = Path(path)
    fmt = fmt or _infer_format(path)
    df = pd.DataFrame(intervals, columns=["start", "end", "label"])
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "lab":
        df.to_csv(path, sep="\t", index=False, header=False)
    else:
        raise ValueError(f"Unknown annotation format {fmt!r}; expected 'csv' or 'lab'.")


def _infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".lab":
        return "lab"
    raise ValueError(
        f"Cannot infer format from extension {ext!r}. Pass fmt='csv' or fmt='lab' explicitly."
    )
