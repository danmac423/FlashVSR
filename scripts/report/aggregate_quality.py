"""Agregacja metryk jakości do tabel rozdziału 8.

Konfiguracje układane są w łańcuch przyrostowy zdefiniowany w manifeście.
Poza wartościami bezwzględnymi skrypt wylicza różnice względem szczebla
poprzedniego, ponieważ to one, a nie same wartości, są przedmiotem omówienia.

Uruchomienie:
    uv run python -m scripts.report.aggregate_quality
"""

from __future__ import annotations

import csv

from scripts.report.common import (
    RESULTS_ROOT,
    load_manifest,
    verify_manifest,
    num,
    signed,
    typst_table,
    write_table,
)

# Metryka, nagłówek, liczba miejsc dziesiętnych, czy niższa wartość jest lepsza
METRICS = [
    ("psnr", "PSNR", 2, False),
    ("ssim", "SSIM", 3, False),
    ("lpips", "LPIPS", 3, True),
    ("niqe", "NIQE", 3, True),
    ("musiq", "MUSIQ", 2, False),
    ("clipiqa", "CLIPIQA", 3, False),
    ("dover", "DOVER", 2, False),
]

DATASET_CAPTION = {
    "YouHQ40": "Metryki jakości dla zbioru YouHQ40, wartości uśrednione po klipach",
    "VideoLQ": "Metryki bezreferencyjne dla zbioru VideoLQ, wartości uśrednione po klipach",
}


def read_average(root: str, dataset: str, label: str) -> dict[str, float | None] | None:
    path = RESULTS_ROOT / root / dataset / f"{label}_merged.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("clip", "").strip().upper() == "AVERAGE":
                out: dict[str, float | None] = {}
                for key, *_ in METRICS:
                    value = row.get(key, "")
                    out[key] = float(value) if value not in (None, "") else None
                return out
    return None


def build_dataset(manifest: dict, dataset: str) -> None:
    quality = manifest["quality"]
    root = quality["root"]

    entries = [(r["file"], r["name"], True) for r in quality["rung"]]
    entries += [(b["file"], b["name"], False) for b in quality["branch"]]

    data = {}
    for label, _, _ in entries:
        avg = read_average(root, dataset, label)
        if avg is not None:
            data[label] = avg

    if not data:
        print(f"  {dataset}: brak plików, pomijam")
        return

    # Metryki obecne w tym zbiorze (pełnoreferencyjnych brak dla VideoLQ).
    present = [
        m for m in METRICS if any(data[label].get(m[0]) is not None for label in data)
    ]

    rows, deltas = [], []
    previous = None
    for label, name, is_rung in entries:
        if label not in data:
            continue
        values = data[label]
        rows.append([name] + [num(values[key], prec) for key, _, prec, _ in present])

        if is_rung and previous is not None:
            delta_row = [name]
            for key, _, prec, lower_better in present:
                a, b = data[previous].get(key), values.get(key)
                if a is None or b is None:
                    delta_row.append("—")
                    continue
                diff = b - a
                mark = "" if abs(diff) > 0 else ""
                delta_row.append(signed(diff, prec) + mark)
            deltas.append(delta_row)
        if is_rung:
            previous = label

    header = ["Konfiguracja"] + [h for _, h, _, _ in present]
    columns = "(auto," + " auto," * len(present)
    columns = columns.rstrip(",") + ")"

    write_table(
        f"jakosc_{dataset.lower()}.typ",
        typst_table(
            caption=DATASET_CAPTION.get(dataset, f"Metryki jakości dla zbioru {dataset}"),
            label=f"tab:jakosc-{dataset.lower()}",
            columns=columns,
            align="(left" + ", right" * len(present) + ")",
            header=header,
            rows=rows,
        ),
    )

    write_table(
        f"jakosc_{dataset.lower()}_delty.typ",
        typst_table(
            caption=(
                f"Zmiana metryk względem szczebla poprzedniego, zbiór {dataset}"
            ),
            label=f"tab:jakosc-{dataset.lower()}-delty",
            columns=columns,
            align="(left" + ", right" * len(present) + ")",
            header=header,
            rows=deltas,
        ),
    )

    # Podsumowanie tekstowe: ile kosztuje pierwszy szczebel, ile wszystkie dalsze.
    rungs = [r["file"] for r in quality["rung"] if r["file"] in data]
    if len(rungs) >= 3:
        print(f"\n  {dataset} — udział pierwszego szczebla w całkowitej zmianie:")
        for key, name, prec, _ in present:
            first = data[rungs[1]].get(key), data[rungs[0]].get(key)
            rest = data[rungs[-1]].get(key), data[rungs[1]].get(key)
            if None in first or None in rest:
                continue
            d_first, d_rest = first[0] - first[1], rest[0] - rest[1]
            total = abs(d_first) + abs(d_rest)
            share = 100 * abs(d_first) / total if total else 0.0
            print(
                f"    {name:>8s}: pierwszy {d_first:+.{prec}f}, "
                f"pozostałe {d_rest:+.{prec}f}  ({share:.0f}% na pierwszym)"
            )


def main() -> None:
    manifest = load_manifest()
    verify_manifest(manifest)
    for dataset in manifest["quality"]["datasets"]:
        print(f"{dataset}")
        build_dataset(manifest, dataset)


if __name__ == "__main__":
    main()
