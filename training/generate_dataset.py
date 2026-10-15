#!/usr/bin/env python3
"""
Generate a synthetic training set for a "glyphgen" model:
plain-English description  ->  Glyph Studio recipe (JSON).

Each example uses the short prompt format the studio sends to models whose name contains "glyphgen":

    GLYPH: <description>
    USE_DRAWING: yes            (optional: animate the user's own drawing)
    BEAT_MS: 469                (optional: sync to a ringtone's beat)
    CURRENT: {...recipe...}     (optional: revise an existing recipe)

and the assistant reply is a compact recipe in exactly the field order and defaults
that index.html's compactRecipe() produces, so examples you save in the studio
can be mixed straight in.

Usage:
    python generate_dataset.py                    # writes data/train.jsonl and data/eval.jsonl
    python generate_dataset.py --n 12000 --seed 7
    python generate_dataset.py --extra my-glyph-examples.jsonl   # mix in examples exported from the studio
"""
import argparse, json, os, random

# ---------------------------------------------------------------- vocabulary
EMOJI = {  # names people say -> emoji
    "cat": "🐱", "kitty": "🐱", "dog": "🐶", "puppy": "🐶", "ghost": "👻", "skull": "💀", "fire": "🔥", "flame": "🔥",
    "lightning bolt": "⚡", "lightning": "⚡", "music note": "🎵", "smiley face": "😀", "happy face": "😀", "winking face": "😉",
    "rocket": "🚀", "pizza": "🍕", "sun": "☀️", "moon": "🌙", "crescent moon": "🌙", "cloud": "☁️", "snowflake": "❄️",
    "flower": "🌸", "alien": "👽", "robot": "🤖", "crown": "👑", "bell": "🔔", "phone": "📱", "envelope": "✉️",
    "game controller": "🎮", "umbrella": "☂️", "coffee cup": "☕", "pumpkin": "🎃", "jack-o-lantern": "🎃", "bat": "🦇",
    "spider": "🕷️", "butterfly": "🦋", "fish": "🐟", "frog": "🐸", "panda": "🐼", "bear": "🐻", "bunny": "🐰",
    "unicorn": "🦄", "dragon": "🐉", "tree": "🌲", "leaf": "🍃", "apple": "🍎", "cherry": "🍒", "cake": "🎂",
    "gift": "🎁", "balloon": "🎈", "trophy": "🏆", "diamond gem": "💎", "key": "🔑", "lock": "🔒", "light bulb": "💡",
    "eye": "👁️", "thumbs up": "👍", "peace sign": "✌️", "wave hand": "👋", "sad face": "😢", "angry face": "😠",
    "heart eyes face": "😍", "sleeping face": "😴", "sunglasses face": "😎", "planet": "🪐", "earth": "🌍",
    "rainbow": "🌈", "umbrella with rain": "☔", "battery": "🔋", "headphones": "🎧", "guitar": "🎸", "car": "🚗",
    "airplane": "✈️", "anchor": "⚓", "bomb": "💣", "poop": "💩", "taco": "🌮", "donut": "🍩", "ice cream": "🍦",
}
SHAPES = {  # names -> (kind, extra fields)
    "heart": ("heart", {}), "star": ("star", {}), "circle": ("circle", {}), "dot": ("circle", {"size": 5}), "ring": ("ring", {}),
    "square": ("square", {}), "box": ("square", {}), "triangle": ("triangle", {}), "diamond": ("diamond", {}),
    "plus sign": ("plus", {}), "cross": ("cross", {}), "x": ("cross", {}), "spiral": ("spiral", {}), "circle of dots": ("dots", {}),
    "line": ("line", {}), "up arrow": ("arrow", {}), "arrow pointing up": ("arrow", {}), "down arrow": ("arrow", {"angle": 180}),
    "right arrow": ("arrow", {"angle": 90}), "left arrow": ("arrow", {"angle": 270}), "diagonal line": ("line", {"angle": 45}),
    "vertical line": ("line", {"angle": 90}),
}
ANIM_PHRASES = {
    "none": ["", "standing still", "that stays still", "glowing steadily"],
    "pulse": ["pulsing", "that pulses", "throbbing", "that grows and shrinks"],
    "beat": ["beating", "that beats like a heartbeat", "with a heartbeat", "thumping"],
    "blink": ["blinking", "that blinks", "flashing on and off", "that flashes"],
    "breathe": ["breathing", "that slowly glows brighter and dimmer", "softly breathing", "that fades in and out"],
    "flicker": ["flickering", "that flickers like a candle", "glitching", "that sputters"],
    "fade_in": ["fading in", "that fades in", "slowly appearing", "that appears gradually"],
    "fade_out": ["fading out", "that fades away", "slowly disappearing", "that vanishes"],
    "spin": ["spinning", "rotating", "that spins clockwise", "turning around"],
    "spin_ccw": ["spinning counterclockwise", "rotating backwards", "that turns anticlockwise"],
    "bounce": ["bouncing", "that bounces", "hopping", "floating up and down"],
    "shake": ["shaking", "that shakes", "wiggling", "trembling"],
    "scroll_left": ["scrolling left", "sliding across from right to left", "that scrolls past"],
    "scroll_right": ["scrolling right", "sliding from left to right"],
    "scroll_up": ["scrolling up", "rising up", "floating upward"],
    "scroll_down": ["falling down", "dropping from the top", "scrolling down"],
    "grow": ["growing", "that grows from nothing", "zooming in", "expanding"],
    "shrink": ["shrinking", "that shrinks away", "zooming out", "getting smaller"],
    "wipe": ["wiping in", "revealed from left to right", "that wipes on"],
}
EFFECTS = {
    "ripple": ["with ripples", "with rings spreading out behind it", "with a ripple behind it", "radiating waves"],
    "sparkle": ["with sparkles", "with twinkling stars around it", "surrounded by sparkles", "with glitter"],
    "rain": ["in the rain", "with rain falling", "with raindrops behind it"],
}
SIZE_WORDS = [("tiny", 5), ("small", 7), ("medium", 9), ("big", 11), ("large", 11), ("huge", 13), ("giant", 13)]
SPEED_WORDS = [("slowly", 0.5, 1.4), ("slow", 0.5, 1.4), ("quickly", 2, 0.8), ("fast", 2, 0.8), ("really fast", 3, 0.7)]
NAMES = ["BELLA", "HI", "HEY", "YES", "NO", "LOL", "OK", "WOW", "SOS", "GG", "GO", "BYE", "MOM", "DAD", "LOVE", "ZOE",
         "MAX", "SAM", "ALEX", "JO", "MIA", "LEO", "WIN", "BRB", "OMG", "DBD", "CALL", "TEXT", "HOME", "2026"]
STARTS = ["", "", "", "make ", "draw ", "show ", "create ", "i want ", "give me ", "design "]


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def layer(kind, anim="none", **kw):
    """Build a layer in compactRecipe() order, omitting defaults."""
    out = {"kind": kind}
    defaults = {"value": "", "size": 9 if kind == "text" else 11, "x": 0, "y": 0, "angle": 0, "brightness": 1}
    for k in ("value", "size", "x", "y", "angle", "brightness"):
        if k in kw and kw[k] != defaults[k]:
            out[k] = round(kw[k], 2) if isinstance(kw[k], float) else kw[k]
    out["anim"] = anim
    for k, d in (("speed", 1), ("start", 0), ("end", 1)):
        if k in kw and kw[k] != d:
            out[k] = round(kw[k], 2) if isinstance(kw[k], float) else kw[k]
    return out


def recipe(name, frames, ms, layers, loop=True):
    r = {"name": name[:40], "frames": frames, "ms": ms}
    if not loop:
        r["loop"] = False
    r["layers"] = layers
    return r


def title(s):
    return " ".join(w.capitalize() for w in s.split())[:40]


# ---------------------------------------------------------------- example builders
def subject(rng):
    if rng.random() < 0.55:
        name = rng.choice(list(EMOJI))
        return name, {"kind": "emoji", "value": EMOJI[name], "size": 12}
    name = rng.choice(list(SHAPES))
    kind, extra = SHAPES[name]
    return name, {"kind": kind, **extra}


def ex_single(rng):
    name, base = subject(rng)
    anim = rng.choice(list(ANIM_PHRASES))
    phrase = rng.choice(ANIM_PHRASES[anim])
    size_word, size = (rng.choice(SIZE_WORDS) if rng.random() < 0.35 else ("", base.get("size", 11)))
    speed_word, speed, ms_mult = (rng.choice(SPEED_WORDS) if rng.random() < 0.3 else ("", 1, 1.0))
    if anim in ("none", "fade_in", "fade_out", "grow", "shrink", "wipe"):
        speed = 1
    frames = 1 if anim == "none" else rng.choice([12, 16, 20, 24])
    ms = int(round(rng.choice([60, 70, 80]) * ms_mult))
    noun = (size_word + " " + name).strip()
    desc = f"{rng.choice(STARTS)}{article(noun)} {noun} {phrase} {speed_word}".strip()
    kw = {k: v for k, v in base.items() if k != "kind"}
    kw["size"] = size
    if speed != 1:
        kw["speed"] = speed
    return desc, recipe(title(name if not phrase else f"{name} {anim.replace('_', ' ')}"), frames, ms, [layer(base["kind"], anim, **kw)])


def ex_effect(rng):
    desc, r = ex_single(rng)
    eff = rng.choice(list(EFFECTS))
    desc = f"{desc} {rng.choice(EFFECTS[eff])}"
    if r["frames"] == 1:
        r["frames"] = 20
    bg = layer(eff, "none", brightness=rng.choice([0.4, 0.5, 0.6]), size=13 if eff == "ripple" else 11)
    r["layers"] = [bg] + r["layers"]
    return desc, r


def ex_text(rng):
    word = rng.choice(NAMES)
    if len(word) <= 2 and rng.random() < 0.5:
        anim = rng.choice(["none", "blink", "pulse", "bounce", "shake"])
        desc = rng.choice([f"show the letters {word}", f"the word {word} {rng.choice(ANIM_PHRASES[anim]) or 'glowing'}", f"display {word}"])
        frames = 1 if anim == "none" else 16
        return desc, recipe(title(word), frames, 80, [layer("text", anim, value=word)])
    desc = rng.choice([f"scroll {word}", f"scroll my name {word}", f"show {word} scrolling across", f"the word {word} sliding by",
                       f"write {word}", f"spell out {word}"])
    frames = 12 + len(word) * 6
    return desc, recipe(title(word), frames, 70, [layer("text", "scroll_left", value=word)])


def ex_countdown(rng):
    n = rng.choice([3, 3, 3, 4, 5, 10]) if rng.random() < 0.9 else 2
    nums = list(range(n, 0, -1))[:5]
    if n == 10:
        nums = [10, 9, 8, 7, 6][:1] + [3, 2, 1]  # keep it short and readable
    k = len(nums)
    anim = rng.choice(["none", "pulse", "fade_out", "grow"])
    layers = []
    for i, v in enumerate(nums):
        s = str(v)
        kw = {"value": s, "size": 11 if len(s) == 1 else 9}
        if i:
            kw["start"] = round(i / k, 2)
        if i < k - 1:
            kw["end"] = round((i + 1) / k, 2)
        layers.append(layer("text", anim, **kw))
    end_fx = rng.random() < 0.4 and len(layers) < 4
    desc = rng.choice([f"countdown from {n}", f"a countdown {' '.join(map(str, nums))}", f"count down {n} to 1", f"{' '.join(map(str, nums))} countdown"])
    if end_fx:
        desc += rng.choice([" then a burst", " ending with a ripple"])
        for l in layers:  # squeeze into the first 80% of the loop
            if "start" in l: l["start"] = round(l["start"] * 0.8, 2)
            l["end"] = round(l.get("end", 1) * 0.8, 2)
        layers.append(layer("ripple", "none", size=13, start=0.8))
    return desc, recipe("Countdown", 10 * len(nums) + (8 if end_fx else 0), 100, layers[:4], loop=False)


def ex_sequence(rng):
    (a, la), (b, lb) = subject(rng), subject(rng)
    if a == b:
        return ex_single(rng)
    style = rng.choice(["then", "fade"])
    k1 = {k: v for k, v in la.items() if k != "kind"}
    k2 = {k: v for k, v in lb.items() if k != "kind"}
    if style == "then":
        desc = f"{article(a)} {a} that turns into {article(b)} {b}" if rng.random() < 0.5 else f"{article(a)} {a} then {article(b)} {b}"
        layers = [layer(la["kind"], "fade_out", end=0.5, **k1), layer(lb["kind"], "fade_in", start=0.5, **k2)]
    else:
        desc = f"{article(a)} {a} that fades in and then fades out"
        layers = [layer(la["kind"], "fade_in", end=0.5, **k1), layer(la["kind"], "fade_out", start=0.5, **k1)]
    return desc, recipe(title(f"{a} to {b}" if style == "then" else a), 24, 70, layers)


def ex_heartbeat_classics(rng):
    n = rng.choice([1, 2, 3])
    words = {1: "once", 2: "twice", 3: "three times"}[n]
    fade = rng.random() < 0.5
    desc = f"a heart that beats {words}" + (", then fades" if fade else "")
    layers = [layer("heart", "beat", speed=n, end=0.7)] + [layer("heart", "fade_out", start=0.7)] if fade else [layer("heart", "beat", speed=n)]
    return desc, recipe("Heartbeat", 24, 70, layers)


def ex_drawing(rng):
    anim = rng.choice([a for a in ANIM_PHRASES if a not in ("none", "scroll_right", "scroll_down")])
    desc = rng.choice([f"make my drawing {rng.choice(ANIM_PHRASES[anim])}", f"animate my drawing so it is {rng.choice(ANIM_PHRASES[anim])}",
                       f"{rng.choice(ANIM_PHRASES[anim])}"])
    layers = [layer("drawing", anim)]
    if rng.random() < 0.3:
        eff = rng.choice(list(EFFECTS)); desc += " " + rng.choice(EFFECTS[eff])
        layers = [layer(eff, "none", brightness=0.5, size=13 if eff == "ripple" else 11)] + layers
    return desc, recipe("My drawing", rng.choice([16, 20, 24]), 70, layers), {"drawing": True}


def ex_beat(rng):
    bms = rng.randint(300, 750)
    per = rng.choice([4, 6, 8])
    beats = rng.choice([1, 2])
    name, base = subject(rng)
    anim = rng.choice(["beat", "pulse", "blink", "bounce"])
    kw = {k: v for k, v in base.items() if k != "kind"}
    kw["speed"] = beats
    desc = rng.choice([f"{article(name)} {name} that {anim}s to the beat", f"{name} {rng.choice(ANIM_PHRASES[anim])} in time with my ringtone",
                       f"sync {article(name)} {name} to the music"])
    return desc, recipe(title(name + " beat"), per * beats, max(20, round(bms / per)), [layer(base["kind"], anim, **kw)]), {"beat_ms": bms}


EDITS = [
    ("make it faster", lambda r: r.update(ms=max(20, round(r["ms"] * 0.6)))),
    ("make it slower", lambda r: r.update(ms=min(400, round(r["ms"] * 1.6)))),
    ("make it bigger", lambda r: [l.update(size=min(13, l.get("size", 11) + 2)) for l in r["layers"]]),
    ("make it smaller", lambda r: [l.update(size=max(3, l.get("size", 11) - 3)) for l in r["layers"]]),
    ("make it dimmer", lambda r: [l.update(brightness=0.5) for l in r["layers"]]),
    ("make it spin", lambda r: r["layers"][-1].update(anim="spin")),
    ("make it bounce", lambda r: r["layers"][-1].update(anim="bounce")),
    ("make it blink", lambda r: r["layers"][-1].update(anim="blink")),
    ("add sparkles", lambda r: r["layers"].insert(0, layer("sparkle", "none", brightness=0.5)) if len(r["layers"]) < 4 else None),
    ("add ripples behind it", lambda r: r["layers"].insert(0, layer("ripple", "none", size=13, brightness=0.5)) if len(r["layers"]) < 4 else None),
    ("don't loop it", lambda r: r.update(loop=False)),
    ("move it up a bit", lambda r: [l.update(y=-2) for l in r["layers"] if l["kind"] not in ("sparkle", "rain", "ripple")]),
]


def ex_refine(rng):
    for _ in range(10):
        desc, cur = ex_single(rng) if rng.random() < 0.6 else ex_effect(rng)
        if cur["frames"] > 1:
            break
    phrase, fn = rng.choice(EDITS)
    new = json.loads(json.dumps(cur))
    fn(new)
    for l in new["layers"]:  # re-normalize to compact form
        l2 = layer(l["kind"], l.get("anim", "none"), **{k: v for k, v in l.items() if k not in ("kind", "anim")})
        l.clear(); l.update(l2)
    if new.get("loop") is True:
        new.pop("loop")
    new = {k: new[k] for k in ("name", "frames", "ms", "loop", "layers") if k in new}
    return phrase, new, {"current": cur}


BUILDERS = [(ex_single, 30), (ex_effect, 14), (ex_text, 10), (ex_countdown, 6), (ex_sequence, 10), (ex_heartbeat_classics, 7),
            (ex_drawing, 8), (ex_beat, 8), (ex_refine, 10)]


def make_example(rng):
    fn = rng.choices([b for b, _ in BUILDERS], weights=[w for _, w in BUILDERS])[0]
    out = fn(rng)
    desc, rec, ctx = (out + ({},))[:3]
    desc = " ".join(desc.split())
    if rng.random() < 0.15:
        desc = desc.capitalize()
    lines = [f"GLYPH: {desc}"]
    if ctx.get("drawing"): lines.append("USE_DRAWING: yes")
    if ctx.get("beat_ms"): lines.append(f"BEAT_MS: {ctx['beat_ms']}")
    if ctx.get("current"): lines.append("CURRENT: " + json.dumps(ctx["current"], ensure_ascii=False, separators=(",", ":")))
    return {"messages": [{"role": "user", "content": "\n".join(lines)},
                         {"role": "assistant", "content": json.dumps(rec, ensure_ascii=False, separators=(",", ":"))}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="training examples")
    ap.add_argument("--eval", type=int, default=300, help="held-out examples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--extra", nargs="*", default=[], help="JSONL files exported from the studio (Export examples)")
    ap.add_argument("--extra-weight", type=int, default=5, help="how many times to repeat each studio example")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    seen, train, evals = set(), [], []
    while len(train) + len(evals) < a.n + a.eval:
        ex = make_example(rng)
        key = ex["messages"][0]["content"]
        if key in seen:
            continue
        seen.add(key)
        (evals if len(evals) < a.eval and rng.random() < 0.05 else train).append(ex)
        if len(train) >= a.n and len(evals) < a.eval:
            evals.append(train.pop())
    for path in a.extra:
        with open(path, encoding="utf-8") as f:
            mine = [json.loads(l) for l in f if l.strip()]
        train.extend(mine * a.extra_weight)
        print(f"added {len(mine)} studio examples x{a.extra_weight} from {path}")
    rng.shuffle(train)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train)} training and {len(evals)} eval examples to {a.out}")


if __name__ == "__main__":
    main()
