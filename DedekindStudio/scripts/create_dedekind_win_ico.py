#!/usr/bin/env python3
"""
Erzeugt dedekind_app_icon_win.ico für die Windows-Taskleiste.
Mehrere Größen inkl. DPI-Varianten (20, 24, 40, …), scharf durch 2x-Render + Downscale.

Aufruf (aus DedekindStudio): python scripts/create_dedekind_win_ico.py
Optional: pip install Pillow
"""
from __future__ import annotations

import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Bitte zuerst installieren: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# LANCZOS für scharfes Downscale (Pillow 10+: Image.Resampling.LANCZOS)
try:
    _resample = Image.Resampling.LANCZOS
except AttributeError:
    _resample = Image.LANCZOS

# Farben wie im SVG: dedekind_app_icon.svg
BG_DARK = (0x1a, 0x2e, 0x24, 255)   # #1a2e24
ACCENT = (0x20, 0xc9, 0x97, 255)     # #20C997 teal
# Für kleine Größen: etwas hellerer Hintergrund, damit es in der Taskleiste nicht schwarz wirkt
BG_LIGHT_SMALL = (0x22, 0x3d, 0x32, 255)


def _draw_at_size(size: int, bg: tuple) -> Image.Image:
    """Zeichnet das Icon bei exakt size x size (ohne Skalierung)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = max(2, int(size * 0.19))
    draw.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=r, fill=bg, outline=None)
    cx, cy = size // 2, size // 2
    font_size = max(6, int(size * 0.55))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
    draw.text((cx, cy), "D", fill=ACCENT, font=font, anchor="mm")
    return img


def draw_icon(size: int) -> Image.Image:
    """Liefert das Icon in der gewünschten Größe; bei kleinen Größen 2x rendern und runterskalieren für Schärfe."""
    use_light_bg = size <= 32
    bg = BG_LIGHT_SMALL if use_light_bg else BG_DARK
    # Kleine Größen: erst in doppelter Auflösung zeichnen, dann mit LANCZOS verkleinern → schärfer in der Taskleiste
    if size <= 48:
        render_size = size * 2
        large = _draw_at_size(render_size, bg)
        return large.resize((size, size), _resample)
    return _draw_at_size(size, bg)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base, "spyder", "images", "dedekind_app_icon_win.ico")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Windows nutzt je nach DPI/Taskleiste 16, 20, 24, 32, 40, 48, 64, 128, 256 – alle liefern, dann kein Skalieren
    sizes = [256, 128, 64, 48, 40, 32, 24, 20, 16]
    images = [draw_icon(s) for s in sizes]
    # Größtes zuerst (ICO-Standard; Windows wählt passende Größe)
    images[0].save(out_path, format="ICO", append_images=images[1:])
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
