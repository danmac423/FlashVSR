"""Wspólne funkcje pomocnicze dla skryptów opracowujących wyniki.

Skrypty w tym katalogu czytają surowe pliki wynikowe wskazane w manifeście
i wytwarzają zagregowane tabele oraz rysunki włączane do pracy. Narzędzia
pomiarowe zapisują wyłącznie wartości surowe, cała agregacja odbywa się tutaj.
"""

from __future__ import annotations

import csv
import statistics
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT  # ścieżki w manifeście są względem korzenia
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "manifest.toml"
TABLES_DIR = REPO_ROOT / "report" / "tables"
FIGURES_DIR = REPO_ROOT / "report" / "figures"


# --------------------------------------------------------------------- manifest


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)


def resolve(relative: str, context: str = "") -> Path:
    """Zamienia ścieżkę z manifestu na bezwzględną, z czytelnym błędem."""
    path = RESULTS_ROOT / relative
    if not path.exists():
        where = f" (wpis '{context}')" if context else ""
        raise FileNotFoundError(
            f"Nie znaleziono pliku wskazanego w manifeście{where}:\n"
            f"  manifest: {MANIFEST_PATH}\n"
            f"  ścieżka w manifeście: {relative}\n"
            f"  szukano w: {path}\n"
            f"Ścieżki w manifeście podaje się względem korzenia repozytorium."
        )
    return path


def verify_manifest(manifest: dict) -> None:
    """Sprawdza wszystkie ścieżki przed rozpoczęciem pracy."""
    missing: list[str] = []
    for spec in manifest.get("performance", []):
        if not (RESULTS_ROOT / spec["path"]).exists():
            missing.append(f"  [{spec['id']}] {spec['path']}")
    for spec in manifest.get("oom", []):
        if not (RESULTS_ROOT / spec["log"]).exists():
            missing.append(f"  [{spec['id']}] {spec['log']}")

    quality = manifest.get("quality", {})
    labels = [r["file"] for r in quality.get("rung", [])]
    labels += [b["file"] for b in quality.get("branch", [])]
    found_any = False
    for dataset in quality.get("datasets", []):
        for label in labels:
            if (RESULTS_ROOT / quality["root"] / dataset / f"{label}_merged.csv").exists():
                found_any = True
    if quality and not found_any:
        missing.append(f"  [quality] nic nie znaleziono w {quality.get('root')}")

    if missing:
        raise SystemExit(
            "Manifest wskazuje na nieistniejące pliki:\n"
            + "\n".join(missing)
            + f"\n\nManifest: {MANIFEST_PATH}"
            + "\nŚcieżki podaje się względem korzenia repozytorium "
            + f"({REPO_ROOT})."
        )


def entry(manifest: dict, entry_id: str) -> dict:
    for item in manifest.get("performance", []):
        if item["id"] == entry_id:
            return item
    raise KeyError(f"Brak wpisu '{entry_id}' w manifeście")


# ------------------------------------------------------------------- wczytywanie


@dataclass
class Run:
    """Pojedynczy przebieg pomiarowy, po ujednoliceniu schematu."""

    label: str
    attn_mode: str
    mask_attn_mode: str
    quantization_mode: str
    input_height: int
    input_width: int
    num_frames: int
    num_spatial_tiles: int
    tile_height: int
    tile_width: int
    inference_time_s: float | None
    peak_vram_mb: float | None
    vram_after_init_mb: float | None
    oom: bool
    source: str = ""
    device: str = ""


def _to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def read_runs(spec: dict) -> list[Run]:
    """Wczytuje plik wynikowy, godząc dwa schematy zapisu.

    Runner wydajnościowy zapisuje wymiary kafla w kolumnach ``tile_height``
    i ``tile_width``, natomiast skrypt badający rozmiar kafla zapisuje pojedynczą
    kolumnę ``tile_size`` oraz znacznik ``oom``. Obie postacie sprowadzane są tu
    do wspólnej reprezentacji.
    """
    path = resolve(spec["path"], spec["id"])
    runs: list[Run] = []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oom = str(row.get("oom", "False")).strip().lower() == "true"

            if row.get("tile_size"):
                tile_h = tile_w = _to_int(row.get("tile_size"))
            else:
                tile_h = _to_int(row.get("tile_height"))
                tile_w = _to_int(row.get("tile_width"))

            runs.append(
                Run(
                    label=row.get("label", ""),
                    attn_mode=row.get("attn_mode", ""),
                    mask_attn_mode=row.get("mask_attn_mode", ""),
                    quantization_mode=row.get("quantization_mode", ""),
                    input_height=_to_int(row.get("input_height")),
                    input_width=_to_int(row.get("input_width")),
                    num_frames=_to_int(row.get("num_frames")),
                    num_spatial_tiles=_to_int(row.get("num_spatial_tiles")),
                    tile_height=tile_h,
                    tile_width=tile_w,
                    inference_time_s=_to_float(row.get("inference_time_s")),
                    peak_vram_mb=_to_float(row.get("peak_vram_mb")),
                    vram_after_init_mb=_to_float(row.get("vram_after_init_mb")),
                    oom=oom,
                    source=spec["id"],
                    device=spec.get("device", ""),
                )
            )
    return runs


# -------------------------------------------------------------------- agregacja


@dataclass
class Aggregate:
    """Wynik zagregowany dla jednej konfiguracji."""

    key: tuple
    runs: list[Run] = field(default_factory=list)

    @property
    def oom(self) -> bool:
        return any(r.oom for r in self.runs)

    @property
    def time_s(self) -> float | None:
        values = [r.inference_time_s for r in self.runs if r.inference_time_s is not None]
        return statistics.median(values) if values else None

    @property
    def peak_mib(self) -> float | None:
        """Maksimum, nie średnia. O mieszczeniu się w budżecie decyduje
        przypadek najgorszy."""
        values = [r.peak_vram_mb for r in self.runs if r.peak_vram_mb is not None]
        return max(values) if values else None

    @property
    def after_init_mib(self) -> float | None:
        values = [r.vram_after_init_mb for r in self.runs if r.vram_after_init_mb is not None]
        return values[0] if values else None

    @property
    def time_per_frame_s(self) -> float | None:
        t = self.time_s
        frames = self.runs[0].num_frames if self.runs else 0
        return t / frames if t and frames else None


def group_by(runs: list[Run], keyfunc) -> dict[tuple, Aggregate]:
    grouped: dict[tuple, Aggregate] = {}
    for run in runs:
        key = keyfunc(run)
        grouped.setdefault(key, Aggregate(key=key)).runs.append(run)
    return grouped


# --------------------------------------------------------------- redundancja


def _tile_coords_fallback(
    height: int, width: int, tile_size: tuple[int, int], overlap: int
) -> list[tuple[int, int, int, int]]:
    """Kopia arytmetyki z ``src.utils.tiling.calculate_spatial_tile_coords``.

    Używana tylko wtedy, gdy raport przygotowywany jest w środowisku bez
    zainstalowanego stosu inferencyjnego. Zgodność obu wersji pilnuje test
    jednostkowy ``tests/test_report_tiling_fallback.py``.
    """
    import math

    tile_w, tile_h = tile_size
    stride_w, stride_h = tile_w - overlap, tile_h - overlap
    coords = []
    for r in range(math.ceil((height - overlap) / stride_h)):
        for c in range(math.ceil((width - overlap) / stride_w)):
            y1, x1 = r * stride_h, c * stride_w
            y2, x2 = min(y1 + tile_h, height), min(x1 + tile_w, width)
            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)
            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)
            coords.append((x1, y1, x2, y2))
    return coords


def tile_geometry(height: int, width: int, tile: int, overlap: int) -> tuple[int, int, float]:
    """Zwraca liczbę kafli, łączną przetworzoną powierzchnię i redundancję.

    Wartości wyznaczane są tą samą funkcją, której używa potok inferencji,
    aby uniknąć rozjechania się opisu z implementacją.
    """
    try:
        from src.utils.tiling import calculate_spatial_tile_coords
    except ImportError:  # środowisko raportowe bez stosu inferencyjnego
        calculate_spatial_tile_coords = _tile_coords_fallback

    # Uwaga: funkcja przyjmuje rozmiar kafla w kolejności (szerokość, wysokość).
    coords = calculate_spatial_tile_coords(height, width, (tile, tile), overlap)
    processed = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in coords)
    frame = height * width
    return len(coords), processed, processed / frame


# ------------------------------------------------------------------ formatowanie


def num(value: float | None, precision: int = 1, dash: str = "-") -> str:
    """Liczba w zapisie polskim, z przecinkiem dziesiętnym."""
    if value is None:
        return dash
    return f"{value:.{precision}f}".replace(".", ",")


def integer(value: float | None, dash: str = "-") -> str:
    if value is None:
        return dash
    return f"{int(round(value)):,}".replace(",", "\u2009")  # spacja półpauzowa


def signed(value: float | None, precision: int = 1, unit: str = "") -> str:
    """Liczba ze znakiem. Wartość zerowa po zaokrągleniu podawana jest bez znaku,
    aby uniknąć zapisów w rodzaju "-0,00"."""
    if value is None:
        return "-"
    rounded = round(value, precision)
    if rounded == 0:
        return f"{0:.{precision}f}".replace(".", ",") + unit
    sign = "+" if rounded > 0 else "\u2212"
    return f"{sign}{abs(rounded):.{precision}f}".replace(".", ",") + unit


# ------------------------------------------------------------- emisja do Typsta


def typst_table(
    caption: str,
    label: str,
    columns: str,
    header: list[str],
    rows: list[list[str]],
    align: str = "left + top",
    text_size: str = "8.5pt",
) -> str:
    """Buduje blok #figure w stylu zgodnym z pozostałymi tabelami pracy."""

    def escape(value: str) -> str:
        # Nawiasy kwadratowe wewnątrz bloku zawartości Typst rozpoczynałyby
        # blok zagnieżdżony, przez co znikałyby z wyniku.
        return value.replace("[", "\\[").replace("]", "\\]")

    def cells(values: list[str]) -> str:
        return ", ".join(f"[{escape(v)}]" for v in values)

    body = ",\n      ".join(cells(r) for r in rows)
    return f"""#figure(
  kind: table,
  caption: [{caption}],
  [
    #set text(size: {text_size})
    #show table.cell.where(y: 0): strong

    #table(
      columns: {columns},
      align: {align},
      stroke: 0.5pt + luma(150),
      fill: (col, row) => if row == 0 {{ luma(230) }} else {{ none }},
      table.header({cells(header)}),

      {body},
    )],
) <{label}>
"""


def write_table(filename: str, content: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  zapisano {path.relative_to(REPO_ROOT)}")
    return path
