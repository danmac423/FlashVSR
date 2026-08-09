"""Wycinanie kadrów porównawczych do rysunku jakościowego.

Wycina ten sam obszar tej samej klatki z kilku konfiguracji potoku i zapisuje
je jako osobne pliki PNG, gotowe do wstawienia w siatkę porównawczą. Materiał LR
jest powiększany metodą najbliższego sąsiada, aby zachować widoczną rozdzielczość
wejścia.

Źródłem może być pojedynczy plik obrazu, katalog klatek albo plik wideo.
Dla pojedynczych obrazów argument --frame jest pomijany.

Przykład:

    python scripts/make_crops.py \\
        --source lr=klatki/lr_042.png \\
        --source baz=klatki/bazowa_042.png \\
        --source kaf=klatki/kafel192_042.png \\
        --source opt=klatki/kafel192_sage_sparge_int8_042.png \\
        --lr-source lr \\
        --crop A=1280,600,400,400 \\
        --crop B=568,288,300,300 \\
        --out rysunki/porownanie

Kolejność argumentów --source wyznacza kolejność kolumn w wygenerowanej siatce.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ACCENT = (57, 255, 20)

# --------------------------------------------------------------------------
# Wczytywanie klatek
# --------------------------------------------------------------------------


def read_frame(path: Path, index: int) -> Image.Image:
    """Wczytuje klatkę o zadanym numerze z pliku wideo lub katalogu obrazów."""
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return Image.open(path).convert("RGB")

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"brak obrazów w katalogu {path}")
        if index >= len(files):
            raise IndexError(f"{path}: żądano klatki {index}, dostępnych {len(files)}")
        return Image.open(files[index]).convert("RGB")

    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("dekodowanie wideo wymaga pakietu av") from exc

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                return frame.to_image().convert("RGB")
    raise IndexError(f"{path}: nie znaleziono klatki {index}")


# --------------------------------------------------------------------------
# Specyfikacja wycinków
# --------------------------------------------------------------------------


@dataclass
class Crop:
    name: str
    x: int
    y: int
    w: int
    h: int


def parse_pair(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"oczekiwano postaci klucz=wartość, otrzymano {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), value.strip()


def build_crop(spec: str) -> Crop:
    """Buduje wycinek z zapisu 'nazwa=x,y,w,h'."""
    name, value = parse_pair(spec)
    parts = [p.strip() for p in value.split(",")]

    if len(parts) != 4:
        raise SystemExit(f"{name}: oczekiwano postaci x,y,w,h")
    x, y, w, h = (int(p) for p in parts)
    return Crop(name, x, y, w, h)


# --------------------------------------------------------------------------
# Główna procedura
# --------------------------------------------------------------------------


def extract(frame: Image.Image, crop: Crop, scale: int, is_lr: bool) -> Image.Image:
    """Wycina obszar; dla materiału LR przelicza współrzędne i powiększa."""
    if is_lr:
        for value, label in ((crop.x, "x"), (crop.y, "y"), (crop.w, "w"), (crop.h, "h")):
            if value % scale:
                raise SystemExit(
                    f"wycinek {crop.name}: {label}={value} nie dzieli się przez {scale}, "
                    "więc nie da się go odwzorować we współrzędnych LR"
                )
        box = (crop.x // scale, crop.y // scale, (crop.x + crop.w) // scale, (crop.y + crop.h) // scale)
        patch = frame.crop(box)
        return patch.resize((crop.w, crop.h), Image.NEAREST)

    box = (crop.x, crop.y, crop.x + crop.w, crop.y + crop.h)
    return frame.crop(box)


def overview(frame: Image.Image, crops: list[Crop], width: int) -> Image.Image:
    """Rysuje pełną klatkę z zaznaczonymi obszarami wycinków."""
    canvas = frame.copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, canvas.width // 400)
    for crop in crops:
        draw.rectangle(
            [crop.x, crop.y, crop.x + crop.w, crop.y + crop.h],
            outline=ACCENT,
            width=line,
        )
    ratio = width / canvas.width
    return canvas.resize((width, round(canvas.height * ratio)), Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", metavar="ETYKIETA=ŚCIEŻKA",
                        help="źródło materiału; kolejność wyznacza kolejność kolumn")
    parser.add_argument("--lr-source", action="append", default=[], metavar="ETYKIETA",
                        help="źródło w rozdzielczości LR, wymagające powiększenia")
    parser.add_argument("--frame", type=int, default=0,
                        help="numer klatki dla katalogów i plików wideo, liczony od zera")
    parser.add_argument("--crop", action="append", metavar="NAZWA=SPEC",
                        help="x,y,w,h we współrzędnych HR")
    parser.add_argument("--out", type=Path, help="katalog wynikowy")
    parser.add_argument("--scale", type=int, default=4, help="krotność powiększenia (domyślnie 4)")
    parser.add_argument("--overview-from", metavar="ETYKIETA",
                        help="źródło dla poglądowej pełnej klatki")
    parser.add_argument("--overview-width", type=int, default=1200)
    opts = parser.parse_args(argv)

    missing = [n for n, v in (("--source", opts.source), ("--crop", opts.crop),
                              ("--out", opts.out)) if v is None]
    if missing:
        raise SystemExit(f"brakuje wymaganych argumentów: {', '.join(missing)}")

    sources = dict(parse_pair(s) for s in opts.source)
    labels = list(sources)
    lr_labels = set(opts.lr_source)
    unknown = lr_labels - set(labels)
    if unknown:
        raise SystemExit(f"--lr-source wskazuje nieznane etykiety: {sorted(unknown)}")

    opts.out.mkdir(parents=True, exist_ok=True)

    frames: dict[str, Image.Image] = {}
    for label, path in sources.items():
        frames[label] = read_frame(Path(path), opts.frame)

    hr_sizes = {label: img.size for label, img in frames.items() if label not in lr_labels}
    if len(set(hr_sizes.values())) > 1:
        raise SystemExit(f"źródła HR mają różne wymiary: {hr_sizes}")
    hr_size = next(iter(hr_sizes.values()))

    for label in sorted(lr_labels):
        lr_size = frames[label].size
        if (lr_size[0] * opts.scale, lr_size[1] * opts.scale) != hr_size:
            raise SystemExit(
                f"źródło {label} ma wymiary {lr_size[0]}×{lr_size[1]}, "
                f"co przy skali {opts.scale} nie odpowiada klatkom HR {hr_size[0]}×{hr_size[1]}"
            )

    if lr_labels:
        ref = sorted(lr_labels)[0]
        lr_size = frames[ref].size
        print(f"wejście LR odczytane z klatki źródła {ref}: {lr_size[0]}×{lr_size[1]}")
    else:
        if hr_size[0] % opts.scale or hr_size[1] % opts.scale:
            raise SystemExit(
                f"klatki HR {hr_size[0]}×{hr_size[1]} nie dzielą się przez skalę {opts.scale}, "
                "a bez źródła LR nie da się wyznaczyć rozmiaru wejścia"
            )
        lr_size = (hr_size[0] // opts.scale, hr_size[1] // opts.scale)
        print(f"wejście LR wyznaczone z klatek HR: {lr_size[0]}×{lr_size[1]}")

    crops = [build_crop(spec) for spec in opts.crop]

    for crop in crops:
        if crop.x < 0 or crop.y < 0 or crop.x + crop.w > hr_size[0] or crop.y + crop.h > hr_size[1]:
            raise SystemExit(f"wycinek {crop.name} wykracza poza klatkę o wymiarach {hr_size}")

    written: list[str] = []
    for crop in crops:
        for label in labels:
            patch = extract(frames[label], crop, opts.scale, label in lr_labels)
            target = opts.out / f"{crop.name}_{label}.png"
            patch.save(target)
            written.append(target.name)

    ref_label = opts.overview_from or next(iter(l for l in labels if l not in lr_labels))
    overview_img = overview(frames[ref_label], crops, opts.overview_width)
    overview_img.save(opts.out / "klatka_pelna.png")
    written.append("klatka_pelna.png")

    manifest = {
        "frame": opts.frame,
        "scale": opts.scale,
        "lr_size": list(lr_size),
        "sources": {label: str(path) for label, path in sources.items()},
        "lr_sources": sorted(lr_labels),
        "crops": [
            {"name": c.name, "x": c.x, "y": c.y, "w": c.w, "h": c.h}
            for c in crops
        ],
        "files": written,
    }
    (opts.out / "crops.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"zapisano {len(written)} plików w {opts.out}")
    for crop in crops:
        print(f"  {crop.name}: {crop.w}×{crop.h} px w ({crop.x}, {crop.y})")

    return 0


if __name__ == "__main__":
    sys.exit(main())