#!/usr/bin/env python3
"""
Synthetic starter data for the pixel model: thousands of captioned 13x13 glyphs drawn with code
(shapes, arrows, letters, digits, faces, weather, objects, and emoji if an emoji font is installed).

Real art from fetch_pixel_art.py / find_glyph_examples.py is better, but there is rarely enough of it.
This teaches the model the basics (where things go, outlines vs filled, positions, sizes) first.

    python synth_glyphs.py                 # -> glyphs/synthetic.jsonl
    python synth_glyphs.py --n 30000 --emoji-font "C:/Windows/Fonts/seguiemj.ttf"
"""
import argparse, json, math, os, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
N, W = 13, 130
yy, xx = np.mgrid[0:N, 0:N]
MASK = np.hypot(xx - 6, yy - 6) <= 6.5
FONT = {"A": "010101111101101", "B": "110101110101110", "C": "011100100100011", "D": "110101101101110", "E": "111100110100111", "F": "111100110100100",
        "G": "011100101101011", "H": "101101111101101", "I": "111010010010111", "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
        "M": "101111111101101", "N": "110101101101101", "O": "010101101101010", "P": "110101110100100", "Q": "010101101110011", "R": "110101110101101",
        "S": "011100010001110", "T": "111010010010010", "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
        "Y": "101101010010010", "Z": "111001010100111", "0": "111101101101111", "1": "010110010010111", "2": "110001010100111", "3": "110001010001110",
        "4": "101101111001001", "5": "111100110001110", "6": "011100111101111", "7": "111001010010010", "8": "111101111101111", "9": "111101111001110",
        "!": "010010010000010", "?": "110001010000010"}
EMOJI = {"cat": "🐱", "dog": "🐶", "ghost": "👻", "skull": "💀", "fire": "🔥", "rocket": "🚀", "pizza": "🍕", "alien": "👽", "robot": "🤖", "crown": "👑",
         "pumpkin": "🎃", "bat": "🦇", "butterfly": "🦋", "fish": "🐟", "frog": "🐸", "panda": "🐼", "bunny": "🐰", "unicorn": "🦄", "apple": "🍎",
         "cherry": "🍒", "cake": "🎂", "gift": "🎁", "balloon": "🎈", "trophy": "🏆", "key": "🔑", "light bulb": "💡", "thumbs up": "👍",
         "peace sign": "✌️", "planet": "🪐", "rainbow": "🌈", "headphones": "🎧", "guitar": "🎸", "car": "🚗", "airplane": "✈️", "anchor": "⚓",
         "bomb": "💣", "taco": "🌮", "donut": "🍩", "ice cream": "🍦", "game controller": "🎮", "flower": "🌸", "mushroom": "🍄", "snail": "🐌",
         "penguin": "🐧", "owl": "🦉", "bee": "🐝", "dinosaur": "🦖", "octopus": "🐙", "teddy bear": "🧸", "diamond gem": "💎", "money bag": "💰"}


def canvas():
    im = Image.new("L", (W, W), 0); return im, ImageDraw.Draw(im)


def finish(im, levels=4):
    a = np.asarray(im.resize((N, N), Image.BOX), float) / 255
    a = np.where(MASK, a, 0)
    q = np.where(a > 0.18, np.round(np.clip(a, 0, 1) * levels) / levels, 0)
    q = np.where((q == 0) & (a > 0.18), 1 / levels, q)
    return (q * 255).astype(int).flatten().tolist()


def poly(d, pts, fill=255, outline=None, width=0):
    d.polygon(pts, fill=fill if not outline else None, outline=outline, width=width)


def star_pts(cx, cy, r, k=5, inner=0.45, rot=-90):
    return [(cx + math.cos(math.radians(rot + i * 180 / k)) * (r if i % 2 == 0 else r * inner), cy + math.sin(math.radians(rot + i * 180 / k)) * (r if i % 2 == 0 else r * inner)) for i in range(2 * k)]


def heart_pts(cx, cy, r):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 80):
        x = 16 * math.sin(t) ** 3; y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * r / 17, cy - y * r / 17 + r * 0.1))
    return pts


def arrow_pts(cx, cy, r, ang):
    base = [(0, -1), (0.75, -0.1), (0.3, -0.1), (0.3, 1), (-0.3, 1), (-0.3, -0.1), (-0.75, -0.1)]
    a = math.radians(ang)
    return [(cx + r * (x * math.cos(a) - y * math.sin(a)), cy + r * (x * math.sin(a) + y * math.cos(a))) for x, y in base]


SIZES = [("tiny", 0.35), ("small", 0.55), ("", 0.8), ("medium", 0.7), ("big", 0.9), ("large", 0.92), ("huge", 1.0)]
PLACES = [("", 0, 0), ("", 0, 0), ("on the left", -0.27, 0), ("on the right", 0.27, 0), ("at the top", 0, -0.27), ("at the bottom", 0, 0.27),
          ("in the top left", -0.2, -0.2), ("in the top right", 0.2, -0.2), ("in the bottom left", -0.2, 0.2), ("in the bottom right", 0.2, 0.2)]
DIRS = {"up": 0, "right": 90, "down": 180, "left": 270, "up right": 45, "down right": 135, "down left": 225, "up left": 315}


def shape(rng):
    kind = rng.choice(["circle", "ring", "square", "box outline", "triangle", "diamond", "star", "star outline", "heart", "heart outline", "plus",
                       "cross", "arrow", "line", "dot", "hexagon", "crescent moon", "sun", "lightning bolt", "drop", "check mark", "smiley face",
                       "sad face", "music note", "house", "tree", "bell", "cloud", "envelope", "battery", "spiral", "grid", "stripes", "checkerboard"])
    sw, sk = rng.choice(SIZES); pw, px, py = rng.choice(PLACES) if kind not in ("grid", "stripes", "checkerboard", "spiral") else ("", 0, 0)
    if pw:
        sk, sw = min(sk, 0.42), (sw if sk <= 0.42 else "small")
    bright = rng.random(); bw = "dim " if bright < 0.15 else ""
    fill = 110 if bw else 255
    im, d = canvas(); r = W / 2 * sk; cx, cy = W / 2 + px * W, W / 2 + py * W
    words = kind; extra = ""
    lw = max(10, int(r * 0.22))
    if kind == "circle": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif kind == "dot": r = W * 0.09; d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill); sw = ""
    elif kind == "ring": d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=lw)
    elif kind == "square": d.rectangle([cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75], fill=fill)
    elif kind == "box outline": d.rectangle([cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75], outline=fill, width=lw)
    elif kind == "triangle":
        dn = rng.choice(["up", "down", "left", "right"]); a = {"up": 0, "right": 90, "down": 180, "left": 270}[dn]
        pts = [(cx + r * math.sin(math.radians(a + t)), cy - r * math.cos(math.radians(a + t))) for t in (0, 120, 240)]; poly(d, pts, fill); extra = f"pointing {dn}"
    elif kind == "diamond": poly(d, [(cx, cy - r), (cx + r * 0.8, cy), (cx, cy + r), (cx - r * 0.8, cy)], fill)
    elif kind == "star": poly(d, star_pts(cx, cy, r), fill)
    elif kind == "star outline": d.polygon(star_pts(cx, cy, r), outline=fill, width=lw)
    elif kind == "heart": poly(d, heart_pts(cx, cy, r), fill)
    elif kind == "heart outline": d.polygon(heart_pts(cx, cy, r), outline=fill, width=lw)
    elif kind == "plus": w = r * 0.32; d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=fill); d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=fill)
    elif kind == "cross": d.line([cx - r * .8, cy - r * .8, cx + r * .8, cy + r * .8], fill=fill, width=int(r * .5)); d.line([cx - r * .8, cy + r * .8, cx + r * .8, cy - r * .8], fill=fill, width=int(r * .5))
    elif kind == "arrow": dn = rng.choice(list(DIRS)); poly(d, arrow_pts(cx, cy, r, DIRS[dn]), fill); extra = f"pointing {dn}"
    elif kind == "line":
        dn = rng.choice(["horizontal", "vertical", "diagonal"]); words = f"{dn} line"
        if dn == "horizontal": d.line([cx - r, cy, cx + r, cy], fill=fill, width=lw)
        elif dn == "vertical": d.line([cx, cy - r, cx, cy + r], fill=fill, width=lw)
        else: d.line([cx - r * .7, cy + r * .7, cx + r * .7, cy - r * .7], fill=fill, width=lw)
    elif kind == "hexagon": poly(d, [(cx + r * math.cos(math.radians(60 * i)), cy + r * math.sin(math.radians(60 * i))) for i in range(6)], fill)
    elif kind == "crescent moon": d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill); d.ellipse([cx - r * .45, cy - r * 1.05, cx + r * 1.35, cy + r * .65], fill=0)
    elif kind == "sun":
        d.ellipse([cx - r * .5, cy - r * .5, cx + r * .5, cy + r * .5], fill=fill)
        for i in range(8): a = math.radians(45 * i); d.line([cx + math.cos(a) * r * .7, cy + math.sin(a) * r * .7, cx + math.cos(a) * r, cy + math.sin(a) * r], fill=fill, width=lw)
    elif kind == "lightning bolt": poly(d, [(cx + r * .2, cy - r), (cx - r * .55, cy + r * .1), (cx - r * .05, cy + r * .1), (cx - r * .25, cy + r), (cx + r * .55, cy - r * .15), (cx + r * .05, cy - r * .15)], fill)
    elif kind == "drop": d.ellipse([cx - r * .6, cy - r * .1, cx + r * .6, cy + r * 1.0], fill=fill); poly(d, [(cx, cy - r), (cx + r * .55, cy + r * .2), (cx - r * .55, cy + r * .2)], fill)
    elif kind == "check mark": d.line([cx - r * .8, cy, cx - r * .2, cy + r * .6, cx + r * .85, cy - r * .7], fill=fill, width=lw + 4, joint="curve")
    elif kind in ("smiley face", "sad face"):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fill, width=lw)
        for ex in (-0.38, 0.38): d.ellipse([cx + ex * r - r * .13, cy - r * .35 - r * .13, cx + ex * r + r * .13, cy - r * .35 + r * .13], fill=fill)
        box = [cx - r * .5, cy - r * .2, cx + r * .5, cy + r * .55] if kind == "smiley face" else [cx - r * .5, cy + r * .25, cx + r * .5, cy + r * .9]
        d.arc(box, 20 if kind == "smiley face" else 200, 160 if kind == "smiley face" else 340, fill=fill, width=lw)
    elif kind == "music note": d.ellipse([cx - r * .6, cy + r * .3, cx, cy + r * .8], fill=fill); d.rectangle([cx - r * .1, cy - r * .8, cx, cy + r * .55], fill=fill); poly(d, [(cx - r * .1, cy - r * .8), (cx + r * .55, cy - r * .5), (cx + r * .55, cy - r * .25), (cx, cy - r * .5)], fill)
    elif kind == "house": poly(d, [(cx, cy - r), (cx + r, cy - r * .05), (cx - r, cy - r * .05)], fill); d.rectangle([cx - r * .7, cy - r * .05, cx + r * .7, cy + r * .9], fill=fill); d.rectangle([cx - r * .18, cy + r * .35, cx + r * .18, cy + r * .9], fill=0)
    elif kind == "tree": poly(d, [(cx, cy - r), (cx + r * .8, cy + r * .45), (cx - r * .8, cy + r * .45)], fill); d.rectangle([cx - r * .15, cy + r * .45, cx + r * .15, cy + r], fill=fill)
    elif kind == "bell": d.chord([cx - r * .7, cy - r * .9, cx + r * .7, cy + r * .5], 180, 360, fill=fill); d.rectangle([cx - r * .7, cy - r * .2, cx + r * .7, cy + r * .5], fill=fill); d.rectangle([cx - r * .9, cy + r * .5, cx + r * .9, cy + r * .65], fill=fill); d.ellipse([cx - r * .18, cy + r * .65, cx + r * .18, cy + r * .95], fill=fill)
    elif kind == "cloud": [d.ellipse([cx + ox * r - rr * r, cy + oy * r - rr * r, cx + ox * r + rr * r, cy + oy * r + rr * r], fill=fill) for ox, oy, rr in ((-.45, .15, .4), (0, -.1, .5), (.45, .15, .4))]; d.rectangle([cx - r * .45, cy + r * .1, cx + r * .45, cy + r * .55], fill=fill)
    elif kind == "envelope": d.rectangle([cx - r, cy - r * .6, cx + r, cy + r * .6], outline=fill, width=lw); d.line([cx - r, cy - r * .6, cx, cy + r * .1, cx + r, cy - r * .6], fill=fill, width=lw)
    elif kind == "battery":
        lvl = rng.choice([("empty", 0), ("low", .25), ("half", .5), ("full", 1.0)]); words = f"battery {lvl[0]}"
        d.rectangle([cx - r * .9, cy - r * .45, cx + r * .75, cy + r * .45], outline=fill, width=lw); d.rectangle([cx + r * .75, cy - r * .2, cx + r * .95, cy + r * .2], fill=fill)
        if lvl[1]: d.rectangle([cx - r * .7, cy - r * .25, cx - r * .7 + r * 1.25 * lvl[1], cy + r * .25], fill=fill)
    elif kind == "spiral":
        pts = [(W / 2 + math.cos(t) * r * t / 12.6, W / 2 + math.sin(t) * r * t / 12.6) for t in np.linspace(0, 12.6, 200)]; d.line(pts, fill=fill, width=lw)
    elif kind == "grid":
        for k in range(0, W, 30): d.line([k + 15, 0, k + 15, W], fill=fill, width=8); d.line([0, k + 15, W, k + 15], fill=fill, width=8)
    elif kind == "stripes":
        dn = rng.choice(["horizontal", "vertical"]); words = f"{dn} stripes"
        for k in range(0, W, 20): d.rectangle([0, k, W, k + 9] if dn == "horizontal" else [k, 0, k + 9, W], fill=fill)
    elif kind == "checkerboard":
        for y in range(N):
            for x in range(N):
                if (x + y) % 2 == 0: d.rectangle([x * 10, y * 10, x * 10 + 9, y * 10 + 9], fill=fill)
    cap = " ".join(w for w in [rng.choice(["", "", "a ", "draw a ", "a glowing "]) + bw + sw, words, extra, pw] if w).strip()
    return " ".join(cap.split()), im


def text_glyph(rng):
    s = rng.choice([rng.choice(list(FONT)), rng.choice(list(FONT)) + rng.choice(list(FONT)), str(rng.randint(0, 99))])
    s = s[:3]
    k = 2 if len(s) == 1 else 1
    im, d = canvas()
    w = (len(s) * 4 - 1) * k * 10; h = 5 * k * 10; ox, oy = (W - w) / 2, (W - h) / 2
    for ci, ch in enumerate(s):
        g = FONT.get(ch, FONT["?"])
        for r in range(5):
            for c in range(3):
                if g[r * 3 + c] == "1":
                    x0 = ox + (ci * 4 + c) * k * 10; y0 = oy + r * k * 10; d.rectangle([x0, y0, x0 + k * 10 - 1, y0 + k * 10 - 1], fill=255)
    kind = "number" if s.isdigit() else "letter" if len(s) == 1 else "letters"
    return rng.choice([f"the {kind} {s}", f"{kind} {s}", f"{s}", f"show {s}"]), im


def emoji_glyph(rng, font):
    name = rng.choice(list(EMOJI)); ch = EMOJI[name]
    img = Image.new("RGBA", (160, 160), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    try:
        d.text((80, 80), ch, font=font, embedded_color=True, anchor="mm")
    except Exception:
        return None
    a = np.asarray(img).astype(float)
    if a[..., 3].max() < 10:
        return None
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) / 255
    v = np.where(a[..., 3] > 128, 0.25 + 0.75 * lum ** 1.2, 0) * 255
    ys, xs = np.nonzero(a[..., 3] > 128)
    crop = Image.fromarray(v.astype(np.uint8)).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    s = rng.choice([0.85, 0.95, 1.0]); side = int(W * s * 0.92)
    crop.thumbnail((side, side), Image.LANCZOS)
    im, _ = canvas(); im.paste(crop, ((W - crop.width) // 2, (W - crop.height) // 2))
    return rng.choice([f"{name}", f"a {name}", f"{name} emoji", f"cute {name}"]), im


def find_emoji_font(path):
    for p in [path, "C:/Windows/Fonts/seguiemj.ttf", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc"]:
        if p and os.path.exists(p):
            for size in (109, 96, 128, 64):
                try:
                    return ImageFont.truetype(p, size)
                except OSError:
                    continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--emoji-font", default=None)
    ap.add_argument("--out", default=str(HERE / "glyphs" / "synthetic.jsonl"))
    a = ap.parse_args()
    rng = random.Random(a.seed); font = find_emoji_font(a.emoji_font)
    print("emoji font:", "found" if font else "none (skipping emoji)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        while n < a.n:
            r = rng.random()
            item = emoji_glyph(rng, font) if (font and r < 0.25) else text_glyph(rng) if r < 0.4 else shape(rng)
            if not item:
                continue
            cap, im = item
            px = finish(im)
            if sum(1 for v in px if v) < 3:
                continue
            f.write(json.dumps({"id": f"syn{n}", "caption": cap, "license": "generated", "frames": [{"ms": 100, "px": px}]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} synthetic glyphs to {a.out}")


if __name__ == "__main__":
    main()
