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

    rungs = [(r["file"], r["name"]) for r in quality["rung"]]
    branches = [(b["file"], b["name"], b["base"]) for b in quality["branch"]]

    data = {}
    for label, _ in rungs:
        avg = read_average(root, dataset, label)
        if avg is not None:
            data[label] = avg
    for label, _, base in branches:
        avg = read_average(root, dataset, label)
        if avg is not None:
            data[label] = avg

    if not data:
        print(f"  {dataset}: brak plików, pomijam")
        return

    present = [m for m in METRICS if any(data[label].get(m[0]) is not None for label in data)]
    header = ["Konfiguracja"] + [h for _, h, _, _ in present]
    columns = "(auto," + " auto," * len(present)
    columns = columns.rstrip(",") + ")"
    align = "(left" + ", right" * len(present) + ")"

    def row(label: str, name: str) -> list[str]:
        return [name] + [num(data[label][key], prec) for key, _, prec, _ in present]

    # --- tabela 1: łańcuch przyrostowy, wartości bezwzględne
    ladder = [(l, n) for l, n in rungs if l in data]
    write_table(
        f"jakosc_{dataset.lower()}.typ",
        typst_table(
            caption=DATASET_CAPTION.get(dataset, f"Metryki jakości dla zbioru {dataset}"),
            label=f"tab:jakosc-{dataset.lower()}",
            columns=columns,
            align=align,
            header=header,
            rows=[row(l, n) for l, n in ladder],
        ),
    )

    # --- tabela 2: zmiany względem konfiguracji poprzedniej
    deltas = []
    for i in range(1, len(ladder)):
        prev_label, curr_label = ladder[i - 1][0], ladder[i][0]
        delta_row = [ladder[i][1]]
        for key, _, prec, _ in present:
            a, b = data[prev_label].get(key), data[curr_label].get(key)
            delta_row.append("-" if a is None or b is None else signed(b - a, prec))
        deltas.append(delta_row)

    write_table(
        f"jakosc_{dataset.lower()}_delty.typ",
        typst_table(
            caption=(f"Zmiana metryk względem konfiguracji poprzedniej, zbiór {dataset}"),
            label=f"tab:jakosc-{dataset.lower()}-delty",
            columns=columns,
            align=align,
            header=header,
            rows=deltas,
        ),
    )

    # --- tabela 3: rozmiary kafla, wraz z konfiguracją odniesienia
    if branches:
        bases = {base for _, _, base in branches if base in data}
        tile_rows = []
        for base in bases:
            name = next(n for l, n in rungs if l == base)
            tile_rows.append(row(base, name.lstrip("+ ")))
        for label, name, _ in branches:
            if label in data:
                tile_rows.append(row(label, name))
        if len(tile_rows) > 1:
            write_table(
                f"jakosc_{dataset.lower()}_kafle.typ",
                typst_table(
                    caption=(
                        f"Metryki jakości dla różnych rozmiarów kafla, zbiór "
                        f"{dataset}. Pozostałe parametry jak w konfiguracji "
                        f"z kafelkowaniem z @tab:jakosc-{dataset.lower()}"
                    ),
                    label=f"tab:jakosc-{dataset.lower()}-kafle",
                    columns=columns,
                    align=align,
                    header=header,
                    rows=tile_rows,
                ),
            )

    # Podsumowanie tekstowe: ile kosztuje pierwszy szczebel, ile wszystkie dalsze.
    rungs = [l for l, _ in ladder]
    if len(rungs) >= 3:
        print(f"\n  {dataset} - udział pierwszego szczebla w całkowitej zmianie:")
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
