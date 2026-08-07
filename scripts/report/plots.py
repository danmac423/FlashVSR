"""Rysunki do rozdziału 8.

Wytwarza dwa rysunki w formacie PDF, włączane w pracy przez #image:

  rozmiar_kafla.pdf   czas i szczytowe zużycie pamięci w funkcji rozmiaru kafla
  e1_czas_pamiec.pdf  czas względem zużycia pamięci dla dwunastu kombinacji E1

Uruchomienie:
    uv run python -m scripts.report.plots
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.report.common import (  # noqa: E402
    FIGURES_DIR,
    REPO_ROOT,
    entry,
    group_by,
    load_manifest,
    read_runs,
)

# Krój i rozmiar dobrane tak, aby rysunek nie odstawał od tekstu pracy.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "figure.constrained_layout.use": True,
    }
)

INK = "#222222"
ACCENT = "#B03A2E"
MUTED = "#7F8C8D"


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {path.relative_to(REPO_ROOT)}")


def plot_tile_size(manifest: dict) -> None:
    spec = entry(manifest, "e2_sweep")
    runs = read_runs(spec)
    groups = group_by(runs, lambda r: r.tile_height)
    budget = manifest["meta"]["budget_mib"]

    tiles = sorted(groups)
    ok = [t for t in tiles if not groups[t].oom]
    oom = [t for t in tiles if groups[t].oom]

    fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(5.5, 4.6), sharex=True)

    # Obszar przepełnienia
    if oom:
        boundary = (max(ok) + min(oom)) / 2
        for ax in (ax_t, ax_m):
            ax.axvspan(boundary, max(tiles) + 12, color=ACCENT, alpha=0.08, lw=0)
            ax.text(
                (boundary + max(tiles) + 12) / 2,
                0.5,
                "przepełnienie pamięci",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                color=ACCENT,
                rotation=90,
            )

    times = [groups[t].time_s for t in ok]
    ax_t.plot(ok, times, "o-", color=INK, lw=1.4, ms=5)
    best = min(ok, key=lambda t: groups[t].time_s)
    ax_t.plot([best], [groups[best].time_s], "o", ms=10, mfc="none", mec=ACCENT, mew=1.6)
    ax_t.annotate(
        "minimum",
        xy=(best, groups[best].time_s),
        xytext=(6, -14),
        textcoords="offset points",
        color=ACCENT,
    )
    ax_t.set_ylabel("czas inferencji [s]")
    ax_t.set_ylim(bottom=min(times) - 6, top=max(times) + 6)

    peaks = [groups[t].peak_mib for t in ok]
    ax_m.plot(ok, peaks, "s-", color=INK, lw=1.4, ms=4.5)
    ax_m.axhline(budget, color=ACCENT, ls="--", lw=1.2)
    ax_m.annotate(
        f"budżet karty ({budget} MiB)",
        xy=(min(tiles), budget),
        xytext=(2, 4),
        textcoords="offset points",
        color=ACCENT,
    )
    ax_m.set_ylabel("szczyt pamięci [MiB]")
    ax_m.set_xlabel("rozmiar kafla [px]")
    ax_m.set_xticks(tiles)
    ax_m.set_ylim(top=budget + 900)

    save(fig, "rozmiar_kafla.pdf")


def plot_e1(manifest: dict) -> None:
    spec = entry(manifest, "e1")
    runs = read_runs(spec)
    groups = group_by(runs, lambda r: (r.attn_mode, r.mask_attn_mode, r.quantization_mode))

    points = [
        (g.peak_mib, g.time_s, key)
        for key, g in groups.items()
        if g.time_s and g.peak_mib
    ]

    # Front niezdominowany: brak konfiguracji jednocześnie szybszej i lżejszej.
    front = [
        p
        for p in points
        if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)
    ]
    front.sort()

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    styles = {
        ("flash", "block_sparse"): ("o", MUTED, "SDPA / blokowo-rzadka"),
        ("sage", "block_sparse"): ("^", MUTED, "SageAttention / blokowo-rzadka"),
        ("flash", "sparse_sage"): ("o", INK, "SDPA / SpargeAttention"),
        ("sage", "sparse_sage"): ("^", INK, "SageAttention / SpargeAttention"),
    }
    seen = set()
    for peak, time, key in points:
        marker, color, legend = styles[(key[0], key[1])]
        ax.plot(
            peak,
            time,
            marker,
            color=color,
            ms=6,
            label=legend if legend not in seen else None,
        )
        seen.add(legend)

    ax.plot(
        [p[0] for p in front],
        [p[1] for p in front],
        "-",
        color=ACCENT,
        lw=1.2,
        alpha=0.8,
        label="front niezdominowany",
    )

    ax.set_xlabel("szczyt pamięci [MiB]")
    ax.set_ylabel("czas inferencji [s]")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    ax.set_ylim(top=max(p[1] for p in points) + 9)

    save(fig, "e1_czas_pamiec.pdf")


def main() -> None:
    manifest = load_manifest()
    print("Rysunki")
    plot_tile_size(manifest)
    plot_e1(manifest)


if __name__ == "__main__":
    main()
