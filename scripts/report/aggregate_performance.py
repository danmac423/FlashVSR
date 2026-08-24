"""Agregacja wyników wydajnościowych do tabel rozdziału 8.

Uruchomienie:
    uv run python -m scripts.report.aggregate_performance
"""

from __future__ import annotations

import sys

from scripts.report.common import (
    Aggregate,
    entry,
    group_by,
    integer,
    load_manifest,
    verify_manifest,
    num,
    read_runs,
    signed,
    tile_geometry,
    typst_table,
    write_table,
)

DENSE = {"flash": "SDPA", "sage": "SageAttention"}
SPARSE = {"block_sparse": "Block Sparse Attention", "sparse_sage": "SpargeAttention", "": "brak"}
QUANT = {
    "none": "brak",
    "int8_weight_only": "INT8 wag",
    "int8_dynamic": "INT8 wag i akt.",
}

# Kolejność wierszy w tabeli E1: najpierw selekcja rzadka, potem jądro gęste,
# na końcu tryb kwantyzacji. Układ ten grupuje warianty różniące się jedną osią.
E1_ORDER = [
    ("flash", "block_sparse"),
    ("sage", "block_sparse"),
    ("flash", "sparse_sage"),
    ("sage", "sparse_sage"),
]
QUANT_ORDER = ["none", "int8_weight_only", "int8_dynamic"]


def build_e1(manifest: dict) -> None:
    spec = entry(manifest, "e1")
    runs = read_runs(spec)
    groups = group_by(runs, lambda r: (r.attn_mode, r.mask_attn_mode, r.quantization_mode))

    ref_key = None
    for key in groups:
        if f"{key[0]}/{key[1]}/{key[2]}" == manifest["meta"]["reference_label"]:
            ref_key = key
    if ref_key is None:
        sys.exit("Nie znaleziono konfiguracji odniesienia w pliku E1")

    ref = groups[ref_key]
    rows, deltas = [], []

    for attn, mask in E1_ORDER:
        for quant in QUANT_ORDER:
            agg = groups.get((attn, mask, quant))
            if agg is None:
                continue
            rows.append(
                [
                    DENSE[attn],
                    SPARSE[mask],
                    QUANT[quant],
                    num(agg.time_s, 2),
                    num(agg.time_per_frame_s, 3),
                    integer(agg.peak_mib),
                ]
            )
            deltas.append(
                [
                    f"{DENSE[attn]} / {SPARSE[mask]} / {QUANT[quant]}",
                    signed(100 * (agg.time_s / ref.time_s - 1), 1, "%"),
                    signed(agg.peak_mib - ref.peak_mib, 0, " MiB"),
                    signed(100 * (agg.peak_mib / ref.peak_mib - 1), 1, "%"),
                ]
            )

    first = runs[0]
    write_table(
        "e1.typ",
        typst_table(
            caption=(
                f"Czas inferencji i szczytowe zużycie pamięci dla dwunastu kombinacji "
                f"mechanizmu uwagi i trybu kwantyzacji "
                f"(wejście {first.input_height} $times$ {first.input_width}, "
                f"kafel {first.tile_height}, {first.num_frames} klatek)"
            ),
            caption_short=(
                "Czas inferencji i szczytowe zużycie pamięci dla dwunastu kombinacji "
                "mechanizmu uwagi i trybu kwantyzacji"
            ),
            label="tab:e1",
            columns="(auto, auto, auto, auto, auto, auto)",
            align="(left, left, left, right, right, right)",
            header=[
                "Jądro gęste",
                "Selekcja rzadka",
                "Kwantyzacja",
                "Czas [s]",
                "s/klatkę",
                "Szczyt [MiB]",
            ],
            rows=rows,
        ),
    )

    write_table(
        "e1_delty.typ",
        typst_table(
            caption="Zmiany względem konfiguracji odniesienia",
            label="tab:e1-delty",
            columns="(auto, auto, auto, auto)",
            align="(left, right, right, right)",
            header=["Konfiguracja", "Czas (wzgl.)", "Pamięć", "Pamięć (wzgl.)"],
            rows=deltas,
        ),
    )

    print("\n  pamięć po inicjalizacji wg trybu kwantyzacji:")
    for quant in QUANT_ORDER:
        vals = {
            g.after_init_mib
            for k, g in groups.items()
            if k[2] == quant and g.after_init_mib is not None
        }
        print(f"    {QUANT[quant]:>16s}: {sorted(vals)}")


def build_e2(manifest: dict) -> None:
    spec = entry(manifest, "e2_sweep")
    runs = read_runs(spec)
    overlap = spec["spatial_overlap"]
    groups = group_by(runs, lambda r: r.tile_height)

    rows = []
    for tile in sorted(groups):
        agg = groups[tile]
        first = agg.runs[0]
        h = first.input_height or runs[0].input_height
        w = first.input_width or runs[0].input_width
        # Wiersze z przepełnieniem nie mają zapisanej rozdzielczości.
        h = h or max(r.input_height for r in runs)
        w = w or max(r.input_width for r in runs)
        count, processed, redundancy = tile_geometry(h, w, tile, overlap)

        if agg.oom:
            rows.append([str(tile), str(count), num(redundancy, 2), "przepełnienie", "—", "—", "—"])
            continue

        rows.append(
            [
                str(tile),
                str(count),
                num(redundancy, 2),
                num(agg.time_s, 2),
                num(agg.time_per_frame_s, 3),
                num(agg.time_s / count, 2),
                integer(agg.peak_mib),
            ]
        )

    write_table(
        "e2.typ",
        typst_table(
            caption=(
                f"Czas inferencji i szczytowe zużycie pamięci w funkcji rozmiaru kafla "
                f"(wejście {max(r.input_height for r in runs)} $times$ "
                f"{max(r.input_width for r in runs)}, zakładka {overlap})"
            ),
            caption_short="Czas inferencji i szczytowe zużycie pamięci w funkcji rozmiaru kafla",
            label="tab:e2",
            columns="(auto, auto, auto, auto, auto, auto, auto)",
            align="(right, right, right, right, right, right, right)",
            header=[
                "Rozmiar kafla",
                "Liczba kafli",
                "Redundancja",
                "Czas [s]",
                "s/klatkę",
                "Czas/kafel [s]",
                "Szczyt [MiB]",
            ],
            rows=rows,
        ),
    )

    print("\n  koszt jednostkowy (czas na przetworzony piksel wejścia):")
    for tile in sorted(groups):
        agg = groups[tile]
        if agg.oom:
            continue
        h = max(r.input_height for r in runs)
        w = max(r.input_width for r in runs)
        _, processed, _ = tile_geometry(h, w, tile, overlap)
        print(f"    kafel {tile}: {agg.time_s / processed:.3e} s/px")


def build_feasibility(manifest: dict) -> None:
    """Tabela granicy wykonalności.

    Zestawia rozmiar kafla niewykonalny w konfiguracji odniesienia z tym samym
    rozmiarem po zastosowaniu badanych technik. Pokazuje przesunięcie granicy,
    a nie samą procentową oszczędność pamięci.
    """
    sweep_spec = entry(manifest, "e2_sweep")
    sweep = group_by(read_runs(sweep_spec), lambda r: r.tile_height)

    spec = entry(manifest, "e2_tile224")
    runs = read_runs(spec)
    agg = Aggregate(key=(spec["id"],), runs=runs)
    tile = runs[0].tile_height

    largest_ok = max(t for t in sweep if not sweep[t].oom)
    rows = [
        [
            "odniesienia",
            str(largest_ok),
            "wykonalna",
            num(sweep[largest_ok].time_s, 2),
            integer(sweep[largest_ok].peak_mib),
        ]
    ]
    if tile in sweep and sweep[tile].oom:
        rows.append(["odniesienia", str(tile), "przepełnienie", "—", "—"])

    dense = DENSE.get(runs[0].attn_mode, runs[0].attn_mode)
    sparse = SPARSE.get(runs[0].mask_attn_mode, runs[0].mask_attn_mode)
    quant = QUANT.get(runs[0].quantization_mode, runs[0].quantization_mode)
    rows.append(
        [
            f"{dense} / {sparse} / {quant}",
            str(tile),
            "wykonalna",
            num(agg.time_s, 2),
            integer(agg.peak_mib),
        ]
    )

    write_table(
        "granica_wykonalnosci.typ",
        typst_table(
            caption=(
                f"Przesunięcie granicy wykonalności: rozmiar kafla niewykonalny "
                f"w konfiguracji odniesienia staje się wykonalny po zastosowaniu "
                f"badanych technik (wejście {runs[0].input_height} $times$ "
                f"{runs[0].input_width}, {runs[0].num_frames} klatek)"
            ),
            caption_short="Przesunięcie granicy wykonalności",
            label="tab:granica",
            columns="(auto, auto, auto, auto, auto)",
            align="(left, right, left, right, right)",
            header=["Konfiguracja", "Kafel", "Wynik", "Czas [s]", "Szczyt [MiB]"],
            rows=rows,
        ),
    )

    budget = manifest["meta"]["budget_mib"]
    print(
        f"\n  kafel {tile} po zastosowaniu technik: {agg.peak_mib:.0f} MiB "
        f"({budget - agg.peak_mib:.0f} MiB poniżej progu), czas {agg.time_s:.1f} s"
    )
    print(f"  ten sam kafel w konfiguracji odniesienia: przepełnienie")


def build_reference(manifest: dict) -> None:
    """Zapotrzebowanie konfiguracji referencyjnej, zmierzone na A100.

    Wyniki z karty docelowej mają charakter binarny (przepełnienie) i nie dają
    się zestawić w tej samej tabeli z wielkościami liczbowymi, dlatego trafiają
    wyłącznie na wyjście, do wykorzystania w tekście.
    """
    total = manifest["meta"]["device_total_mib"]

    a100 = {}
    for spec in manifest["performance"]:
        if not spec["id"].startswith("ref_a100"):
            continue
        runs = read_runs(spec)
        a100[(runs[0].input_height, runs[0].input_width)] = Aggregate(key=(spec["id"],), runs=runs)

    rows = []
    for res in sorted(a100, key=lambda r: r[0] * r[1]):
        agg = a100[res]
        rows.append(
            [
                f"{res[0]} $times$ {res[1]}",
                integer(agg.peak_mib),
                num(agg.time_s, 2),
                num(agg.peak_mib / total, 2) + "$times$",
            ]
        )

    write_table(
        "referencja.typ",
        typst_table(
            caption="Zapotrzebowanie konfiguracji bazowej, zmierzone na akceleratorze A100",
            caption_short="Zapotrzebowanie konfiguracji bazowej",
            label="tab:referencja",
            columns="(auto, auto, auto, auto)",
            align="(right, right, right, right)",
            header=[
                "Wejście",
                "Szczyt [MiB]",
                "Czas [s]",
                "Krotność pojemności RTX 3080",
            ],
            rows=rows,
        ),
    )

    print("\n  przepełnienia na karcie docelowej (do wykorzystania w tekście):")
    for o in manifest.get("oom", []):
        print(
            f"    {o['input_height']}x{o['input_width']}: wolne "
            f"{o['free_mib']:.0f} MiB, nieudana alokacja {o['failed_alloc_mib']:.0f} MiB"
        )

    if len(a100) == 2:
        small, big = sorted(a100, key=lambda r: r[0] * r[1])
        area = (big[0] * big[1]) / (small[0] * small[1])
        peak = a100[big].peak_mib / a100[small].peak_mib
        const = a100[small].after_init_mib or 0
        variable = (a100[big].peak_mib - const) / (a100[small].peak_mib - const)
        print("\n  charakter wzrostu bez kafelkowania:")
        print(f"    powierzchnia   x{area:.2f}")
        print(f"    szczyt         x{peak:.2f}  (składowa stała {const:.0f} MiB)")
        print(f"    część zmienna  x{variable:.2f}")


def main() -> None:
    manifest = load_manifest()
    verify_manifest(manifest)
    print("E1 - mechanizm uwagi i kwantyzacja")
    build_e1(manifest)
    print("\nE2 - rozmiar kafla")
    build_e2(manifest)
    print("\nGranica wykonalności")
    build_feasibility(manifest)
    print("\nKonfiguracja referencyjna")
    build_reference(manifest)


if __name__ == "__main__":
    main()
