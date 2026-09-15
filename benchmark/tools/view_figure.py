"""Composite a source image and several view_classify.py maps into one figure.

A classification map read alone says little: the eye needs the photograph beside
it to know whether a red region is a rope, a step or the sky's edge. Panels
share one geometry and one legend so a difference between two of them is a
difference in the reconstruction, never in the rendering.
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LEGEND = [("hit", (41, 173, 97)), ("wrong", (217, 61, 140)),
          ("missed", (230, 56, 51)), ("extra", (242, 199, 64)),
          ("void", (26, 28, 33))]
BG = (18, 19, 23)
FG = (232, 234, 238)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--panel", action="append", default=[],
                    help="label=path.png, repeatable; split at the LAST = so labels may contain one")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--width", type=int, default=760, help="per-panel width")
    ap.add_argument("--title", default="")
    ap.add_argument("--no-legend", action="store_true",
                    help="for panels that are cloud renders, not classification maps")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    panels = [("source image", a.photo)] + [tuple(p.rsplit("=", 1)) for p in a.panel]
    ims = []
    for label, path in panels:
        im = Image.open(path).convert("RGB")
        h = int(round(im.height * a.width / im.width))
        ims.append((label, im.resize((a.width, h), Image.LANCZOS)))

    pw, ph = a.width, ims[0][1].height
    cols = min(a.cols, len(ims))
    rows = (len(ims) + cols - 1) // cols
    pad, bar, top = 12, 30, 46 if a.title else 10
    legend_h = 0 if a.no_legend else 34
    W = cols * pw + (cols + 1) * pad
    H = top + rows * (ph + bar) + (rows + 1) * pad + legend_h

    out = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(out)
    if a.title:
        d.text((pad, 14), a.title, fill=FG, font=font(26))

    for i, (label, im) in enumerate(ims):
        r, c = divmod(i, cols)
        x = pad + c * (pw + pad)
        y = top + pad + r * (ph + bar + pad)
        d.text((x + 2, y), label, fill=FG, font=font(15))
        out.paste(im, (x, y + bar))

    if not a.no_legend:
        y = H - legend_h + 8
        x = pad
        f = font(14)
        for name, col in LEGEND:
            d.rectangle([x, y, x + 14, y + 14], fill=col)
            d.text((x + 20, y - 1), name, fill=FG, font=f)
            x += 20 + int(d.textlength(name, font=f)) + 22
    out.save(a.out, quality=92)
    print(f"{a.out}  {W}x{H}  {len(ims)} panels")


if __name__ == '__main__':
    main()
