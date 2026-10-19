#!/usr/bin/env python3
"""
Turn downloaded pixel art (raw/manifest.jsonl from fetch_pixel_art.py) into 13x13 Glyph Matrix frames.

Uses the same steps as the studio's "Image or GIF to glyph" panel: detect and remove the background,
trim empty edges, fit into the 137-LED circle, sample sharply for pixel art, then group the picture's
tones into a few shades (1-D k-means) so each region keeps its own brightness.

Outputs (in glyphs/):
  glyphs.jsonl        one row per image/GIF: caption, tags, license, credit, frames [{ms, px[169]}]
  patterns/*.json     the same glyphs in Glyph Studio's pattern format (open them with Import...)
  contact_sheet.png   a preview grid, to spot bad conversions quickly
  rejected.jsonl      what was skipped and why

Usage:
    pip install pillow numpy
    python convert_to_glyphs.py
    python convert_to_glyphs.py --shades 3 --max-frames 32 --allow-photos
"""
import argparse, json, re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

HERE = Path(__file__).resolve().parent
N, C, R, W = 13, 6, 6.5, 130
yy, xx = np.mgrid[0:N, 0:N]
MASK = (np.hypot(xx - C, yy - C) <= R)
assert MASK.sum() == 137
STOP = set("pixel art pixelart sprite sprites icon icons png gif image file 8bit 8-bit 16bit 16-bit animated animation the a of and with by".split())


def frames_of(path, max_frames, step):
    im = Image.open(path)
    out = []
    for i, fr in enumerate(ImageSequence.Iterator(im)):
        if i % step:
            if out:
                out[-1][1] += fr.info.get("duration", 100)
            continue
        f = fr.convert("RGBA")
        if max(f.size) > 512:
            f.thumbnail((512, 512), Image.LANCZOS)
        out.append([f, fr.info.get("duration", 100) or 100])
        if len(out) >= max_frames:
            break
    return out


def detect_bg(a):
    """a: HxWx4 uint8. Returns ('alpha', None), ('color', rgb) or (None, None) for busy edges."""
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(int)
    if (edge[:, 3] < 128).mean() > 0.5:
        return "alpha", None
    rgb = np.median(edge[:, :3], axis=0)
    close = (np.abs(edge[:, :3] - rgb).sum(1) < 60).mean()
    return ("color", rgb) if close > 0.6 else (None, None)


def fg_mask(a, bg):
    m = a[..., 3] >= 128
    if bg[0] == "color":
        m &= np.abs(a[..., :3].astype(int) - bg[1]).sum(-1) >= 70
    return m


def to_glyph(frames, shades=4, floor=0.18, contrast=1.3, zoom=1.0, sampling="auto"):
    arrs = [np.asarray(f) for f, _ in frames]
    bg = detect_bg(arrs[0])
    masks = [fg_mask(a, bg) for a in arrs]
    union = np.zeros(arrs[0].shape[:2], bool)
    for m in masks:
        if m.shape == union.shape:
            union |= m
    if not union.any():
        return None, "empty after background removal"
    ys, xs = np.where(union)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    bw, bh = box[2] - box[0], box[3] - box[1]
    small = max(arrs[0].shape[:2]) <= 96
    sharp = sampling == "sharp" or (sampling == "auto" and small)
    k = (W * 0.92 * zoom) / max(bw, bh)
    w, h = max(1, round(bw * k)), max(1, round(bh * k))
    raw = []
    for (f, ms), a in zip(frames, arrs):
        crop = f.crop(box).resize((w, h), Image.NEAREST if sharp else Image.LANCZOS)
        canvas = Image.new("RGBA", (W, W), (0, 0, 0, 0)); canvas.paste(crop, ((W - w) // 2, (W - h) // 2))
        c = np.asarray(canvas); m = fg_mask(c, bg)
        lum = (0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]) / 255
        cov = m.reshape(N, 10, N, 10).mean((1, 3))
        lsum = (lum * m).reshape(N, 10, N, 10).sum((1, 3))
        L = np.where(cov > 0, lsum / np.maximum(cov * 100, 1), 0)
        L = np.clip((L - 0.5) * contrast + 0.5, 0, 1)
        v = np.maximum(floor, L) * (cov > 0)
        v = np.where(cov >= 0.4, v, 0) if sharp else v * np.minimum(1, cov * 1.6)
        raw.append(v * MASK)
    lit = np.concatenate([r[r > 0] for r in raw]) if raw else np.array([])
    if lit.size == 0:
        return None, "nothing lit"
    uniq = np.unique(np.round(lit * 200) / 200)
    kk = min(shades, len(uniq))
    cent = uniq[np.round(np.linspace(0, len(uniq) - 1, kk)).astype(int)].astype(float)
    for _ in range(20):
        idx = np.abs(lit[:, None] - cent[None]).argmin(1)
        cent = np.array([lit[idx == j].mean() if (idx == j).any() else cent[j] for j in range(kk)])
    cent = np.unique(np.round(cent, 3))
    levels = np.array([255] if len(cent) == 1 else np.round(255 * (0.2 + 0.8 * np.arange(len(cent)) / (len(cent) - 1))))
    out = []
    for (f, ms), v in zip(frames, raw):
        q = np.zeros((N, N), int)
        on = v > 0
        q[on] = levels[np.abs(v[on][:, None] - cent[None]).argmin(1)]
        out.append({"ms": int(min(5000, max(20, ms))), "px": q.flatten().tolist()})
    return out, None


def looks_like_photo(path, frames):
    f = frames[0][0]
    if len(frames) > 1 or max(f.size) <= 128:
        return False
    t = f.convert("RGB").copy(); t.thumbnail((96, 96))
    return len(t.getcolors(96 * 96) or []) > 1500


def caption(row):
    words = []
    for src in [row.get("query", "")] + [row.get("title", "")] + row.get("tags", [])[:6]:
        for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", src.lower()):
            if w not in STOP and len(w) > 1 and w not in words and not re.match(r"^(img|dsc|file|frame|copy|final|v\d)$", w):
                words.append(w)
    return " ".join(words[:8]) or "pixel art"


def contact_sheet(rows, path, cols=20):
    rows = rows[:400]; cell = 44
    sheet = Image.new("RGB", (cols * cell, max(1, (len(rows) + cols - 1) // cols) * cell), (8, 8, 9)); d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        px = r["frames"][0]["px"]; ox, oy = (i % cols) * cell + 2, (i // cols) * cell + 2
        for j, v in enumerate(px):
            if MASK.flat[j]:
                x, y = ox + (j % N) * 3, oy + (j // N) * 3
                g = 28 + int(v * 0.89) if v else 28
                d.rectangle([x, y, x + 1, y + 1], fill=(g, g, g - 4 if g > 4 else g))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "raw"))
    ap.add_argument("--out", default=str(HERE / "glyphs"))
    ap.add_argument("--shades", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--step", type=int, default=1, help="use every Nth GIF frame")
    ap.add_argument("--allow-photos", action="store_true", help="keep images that look like photos instead of pixel art")
    ap.add_argument("--min-lit", type=int, default=8); ap.add_argument("--max-lit", type=int, default=128)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    (out / "patterns").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (raw / "manifest.jsonl").open(encoding="utf-8")]
    kept, rejected = [], []
    for r in rows:
        try:
            frames = frames_of(raw / r["file"], a.max_frames, a.step)
            if not a.allow_photos and looks_like_photo(raw / r["file"], frames):
                rejected.append({"file": r["file"], "why": "looks like a photo"}); continue
            g, why = to_glyph(frames, shades=a.shades)
            if g is None:
                rejected.append({"file": r["file"], "why": why}); continue
            lit = max(sum(1 for v in f["px"] if v) for f in g)
            if not (a.min_lit <= lit <= a.max_lit):
                rejected.append({"file": r["file"], "why": f"{lit} LEDs lit"}); continue
            if len(g) > 1 and all(f["px"] == g[0]["px"] for f in g):
                g = g[:1]
            gid = r["sha256"][:12]
            row = {"id": gid, "caption": caption(r), "query": r.get("query", ""), "title": r.get("title", ""), "tags": r.get("tags", []),
                   "license": r.get("license"), "creator": r.get("creator", ""), "page": r.get("page", ""), "frames": g}
            kept.append(row)
            pattern = {"format": "glyphstudio/1", "device": "DEVICE_25111p", "size": 13, "ledCount": 137, "name": row["caption"][:40],
                       "loop": True, "totalMs": sum(f["ms"] for f in g), "audio": None, "frames": g}
            (out / "patterns" / f"{gid}.json").write_text(json.dumps(pattern))
        except Exception as e:
            rejected.append({"file": r["file"], "why": f"error: {e}"})
    with (out / "glyphs.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    contact_sheet(kept, out / "contact_sheet.png")
    anim = sum(1 for r in kept if len(r["frames"]) > 1)
    print(f"Converted {len(kept)} of {len(rows)} ({anim} animated). Skipped {len(rejected)} (see rejected.jsonl).")
    print(f"Preview: {out / 'contact_sheet.png'}   Training data: {out / 'glyphs.jsonl'}")


if __name__ == "__main__":
    main()
