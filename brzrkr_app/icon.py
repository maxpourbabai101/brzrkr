"""Generate the BRZRKR icon programmatically with PIL.

A circular black emblem with a stylised "B" mark in crimson, ringed
by a gothic ornament. Saved to ``assets/BRZRKR_icon.png`` (1024×1024)
on first call; subsequent calls just load it. Also generates
``assets/BRZRKR_icon.icns`` if iconutil is available (macOS only).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ICON_PNG = ASSETS / "BRZRKR_icon.png"
ICON_ICNS = ASSETS / "BRZRKR_icon.icns"

VOID = (6, 4, 10, 255)
BLOOD = (139, 10, 20, 255)
BLOOD_HI = (196, 30, 42, 255)
BONE = (230, 220, 200, 255)
IRON = (44, 38, 48, 255)
EMBER = (229, 72, 24, 255)


def generate(force: bool = False) -> Path:
    """Render and save the icon at 1024×1024. Returns the PNG path."""
    if ICON_PNG.exists() and not force:
        return ICON_PNG

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed — skipping icon generation.")
        return ICON_PNG

    ASSETS.mkdir(parents=True, exist_ok=True)

    S = 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer ring (crimson) on void background — sharp circle.
    pad = 24
    draw.ellipse([pad, pad, S - pad, S - pad], fill=VOID,
                 outline=BLOOD_HI, width=10)
    # Inner ring (iron)
    pad2 = 64
    draw.ellipse([pad2, pad2, S - pad2, S - pad2], outline=IRON, width=3)
    # Tighter blood ring with hairline
    pad3 = 96
    draw.ellipse([pad3, pad3, S - pad3, S - pad3], outline=BLOOD, width=2)

    # Decorative cross marks at 0/90/180/270
    cx, cy = S // 2, S // 2
    for ang_deg in (0, 90, 180, 270):
        # Sharp triangular notches pointing inward.
        import math
        a = math.radians(ang_deg)
        rx = (S / 2 - pad)
        x1, y1 = cx + rx * math.cos(a), cy + rx * math.sin(a)
        x2, y2 = cx + (rx - 40) * math.cos(a), cy + (rx - 40) * math.sin(a)
        draw.line([(x1, y1), (x2, y2)], fill=BLOOD_HI, width=6)

    # Center "B" with sword-slash
    # Try several serif fonts; fall back to default.
    font_path = _find_font([
        "/System/Library/Fonts/Supplemental/Trajan.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Optima.ttc",
        "/Library/Fonts/Optima.ttc",
        "/System/Library/Fonts/Times.ttc",
    ])
    try:
        font = ImageFont.truetype(font_path, 540) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    text = "B"
    # Anchor the text at the center using bounding box.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cx - tw // 2 - bbox[0]
        ty = cy - th // 2 - bbox[1] - 30
    except Exception:
        tw = th = 400
        tx, ty = cx - 200, cy - 220

    # Shadow / depth
    draw.text((tx + 4, ty + 4), text, fill=(0, 0, 0, 200), font=font)
    # Body
    draw.text((tx, ty), text, fill=BONE, font=font)

    # Diagonal slash through the "B" — Berserk/Vagabond sword cut
    draw.line([(cx - 280, cy + 200), (cx + 280, cy - 200)],
              fill=EMBER, width=14)
    draw.line([(cx - 280, cy + 200), (cx + 280, cy - 200)],
              fill=BLOOD_HI, width=4)

    # Small text below: "BRZRKR"
    try:
        sub_font = ImageFont.truetype(font_path, 56) if font_path else ImageFont.load_default()
        sub_text = "BRZRKR"
        try:
            sb = draw.textbbox((0, 0), sub_text, font=sub_font)
            sw = sb[2] - sb[0]
            draw.text((cx - sw // 2 - sb[0], cy + 240),
                       sub_text, fill=BLOOD_HI, font=sub_font)
        except Exception:
            pass
    except Exception:
        pass

    img.save(ICON_PNG, "PNG")
    logger.info("Generated icon: %s", ICON_PNG)
    return ICON_PNG


def _find_font(candidates) -> str | None:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def to_icns() -> Path | None:
    """Convert the PNG to .icns using macOS iconutil. Returns the path
    or None if iconutil isn't available."""
    if not ICON_PNG.exists():
        generate()
    if sys.platform != "darwin":
        return None
    if ICON_ICNS.exists():
        return ICON_ICNS
    try:
        from PIL import Image
    except ImportError:
        return None
    iconset = ASSETS / "BRZRKR_icon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    src = Image.open(ICON_PNG)
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        out = iconset / f"icon_{s}x{s}.png"
        src.resize((s, s), Image.LANCZOS).save(out, "PNG")
        out2 = iconset / f"icon_{s // 2}x{s // 2}@2x.png"
        # @2x variant for retina
        if s // 2 >= 16:
            src.resize((s, s), Image.LANCZOS).save(out2, "PNG")
    try:
        subprocess.run(["iconutil", "-c", "icns", str(iconset),
                         "-o", str(ICON_ICNS)], check=True)
        return ICON_ICNS
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def set_window_icon(window) -> None:
    """Try to set the Tk window icon. Best-effort, no-op if unsupported."""
    try:
        path = generate()
        if not path.exists():
            return
        from PIL import Image, ImageTk
        img = Image.open(path).resize((64, 64), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        window._brzrkr_icon = photo  # keep ref
        window.iconphoto(True, photo)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not set window icon: %s", exc)
