#!/usr/bin/env python3
"""
Train "glyphdraw", a small text-to-glyph model, FROM SCRATCH on 13x13 glyphs.

It is tiny (about 5 M parameters by default), so it trains in roughly 20-60 minutes on a 24 GB GPU,
or longer on an 8 GB card / CPU. The text side uses a pre-trained sentence encoder, so the model
understands descriptions it has not seen word-for-word.

Data (any mix; real art is weighted up):
    glyphs/synthetic.jsonl        python synth_glyphs.py
    glyphs/glyphs.jsonl           python fetch_pixel_art.py / find_glyph_examples.py, then convert_to_glyphs.py
    my-glyph-examples.jsonl       Export from the studio is recipe data, not pixels - skip it here

    pip install torch numpy pillow sentence-transformers
    python synth_glyphs.py
    python train_pixel_model.py                          # writes pixel-model/glyphdraw.pt + samples.png
    python train_pixel_model.py --epochs 40 --d 384 --layers 8      # bigger, for a 24 GB GPU and lots of data
    python train_pixel_model.py --encoder hash           # no download (worse at synonyms)

Then serve it to the studio:  python pixel_server.py
"""
import argparse, json, math, random, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import pixel_model as pm

HERE = Path(__file__).resolve().parent
FLIP_WORDS = {"left": "right", "right": "left"}
TEXTY = re.compile(r"\b(letter|letters|number|text|word|show)\b|\b[A-Z0-9]{1,3}\b")


def load_rows(paths, real_weight):
    items = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  (missing {p}, skipped)"); continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        w = 1 if "synthetic" in p.name else real_weight
        for r in rows:
            frames = r.get("frames") or []
            # Stills: the frame. Animations: up to 4 evenly spaced frames, all with the same caption.
            pick = frames if len(frames) <= 4 else [frames[int(i * (len(frames) - 1) / 3)] for i in range(4)]
            for f in pick:
                if len(f.get("px", [])) == 169 and any(f["px"]):
                    items += [(r.get("caption") or "glyph", f["px"])] * w
        print(f"  {p.name}: {len(rows)} glyphs (weight {w})")
    return items


def augment(caption, px, rng):
    """Shift by one LED, or mirror (fixing left/right in the caption). Never mirror text."""
    a = np.array(px).reshape(13, 13)
    if rng.random() < 0.3 and not TEXTY.search(caption):
        a = a[:, ::-1]
        caption = re.sub(r"\b(left|right)\b", lambda m: FLIP_WORDS[m.group(1)], caption)
    if rng.random() < 0.4:
        dy, dx = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        b = np.roll(a, (dy, dx), (0, 1))
        if b.flatten()[~pm.MASK].sum() == 0 and (b > 0).sum() == (a > 0).sum():  # nothing fell off the circle
            a = b
    a = a.flatten(); a[~pm.MASK] = 0
    return caption, a.tolist()


def preview(model, enc, prompts, device, path, n=4, guidance=3.0):
    cell = 56; im = Image.new("RGB", (n * cell + 150, len(prompts) * cell), (8, 8, 9)); d = ImageDraw.Draw(im)
    for i, p in enumerate(prompts):
        t = torch.tensor(enc.encode([p])[0], device=device)
        toks = model.sample(t, n=n, guidance=guidance).cpu().numpy()
        d.text((4, i * cell + cell // 2 - 6), p[:22], fill=(220, 220, 220))
        for k, tk in enumerate(toks):
            px = pm.tokens_to_px(tk); ox, oy = 150 + k * cell, i * cell
            for j, v in enumerate(px):
                if pm.MASK[j]:
                    g = 30 + int(v * 0.88) if v else 30; x, y = ox + 4 + (j % 13) * 4, oy + 4 + (j // 13) * 4
                    d.rectangle([x, y, x + 2, y + 2], fill=(g, g, g - 3 if g > 3 else g))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[str(HERE / "glyphs" / "synthetic.jsonl"), str(HERE / "glyphs" / "glyphs.jsonl")])
    ap.add_argument("--real-weight", type=int, default=4, help="repeat real (non-synthetic) glyphs this many times")
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "hash"])
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--cfg-drop", type=float, default=0.1, help="share of steps trained without the prompt (enables guidance)")
    ap.add_argument("--out", default=str(HERE / "pixel-model"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    print("Loading data:")
    items = load_rows(a.data, a.real_weight)
    if len(items) < 200:
        raise SystemExit("Not enough glyphs. Run synth_glyphs.py (and the fetch/convert scripts) first.")
    rng.shuffle(items)
    n_val = max(200, len(items) // 50)
    val, train = items[:n_val], items[n_val:]
    print(f"  {len(train)} training / {len(val)} validation examples")

    enc = pm.make_encoder(a.encoder)
    print(f"Text encoder: {enc.name} ({enc.dim}-d)")
    cache = {}
    def embed(caps):
        new = [c for c in set(caps) if c not in cache]
        for i in range(0, len(new), 1024):
            for c, e in zip(new[i:i + 1024], enc.encode(new[i:i + 1024])):
                cache[c] = e
        return np.stack([cache[c] for c in caps])

    dev = torch.device(a.device)
    model = pm.GlyphDraw(enc.dim, a.d, a.layers, a.heads).to(dev)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def batch_tensors(rows, aug):
        caps, toks = [], []
        for c, px in rows:
            if aug:
                c, px = augment(c, px, rng)
            caps.append(c); toks.append(pm.px_to_tokens(px))
        return torch.tensor(embed(caps), device=dev), torch.tensor(np.stack(toks), device=dev)

    def loss_on(text, tok, drop):
        logits = model(text, tok[:, :-1], drop_mask=drop)          # predict LED k from the prompt + LEDs < k
        return F.cross_entropy(logits.reshape(-1, pm.LEVELS), tok.reshape(-1))

    prompts = ["a heart", "a ghost", "smiley face", "arrow pointing left", "the number 7", "a star in the top right", "lightning bolt", "a cat"]
    best, step, t0 = float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), a.batch):
            text, tok = batch_tensors(train[i:i + a.batch], aug=True)
            drop = torch.rand(len(tok), device=dev) < a.cfg_drop
            with torch.autocast(dev.type, dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16, enabled=use_amp):
                loss = loss_on(text, tok, drop)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(tok); step += 1
        model.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(val), a.batch):
                text, tok = batch_tensors(val[i:i + a.batch], aug=False)
                vl += loss_on(text, tok, None).item() * len(tok)
        vl /= len(val)
        flag = ""
        if vl < best:
            best = vl; flag = "  (saved)"
            pm.save(out / "glyphdraw.pt", model, enc.name, {"val_loss": vl, "epoch": ep, "examples": len(train), "data": a.data})
        print(f"epoch {ep:3d}/{a.epochs}  train {tot / len(train):.4f}  val {vl:.4f}  {time.time() - t0:5.0f}s{flag}")
        if ep % 5 == 0 or ep == a.epochs:
            preview(model, enc, prompts, dev, out / "samples.png")
    model, _, _ = pm.load(out / "glyphdraw.pt", dev)
    preview(model, enc, prompts, dev, out / "samples.png")
    print(f"\nBest validation loss {best:.4f}. Model: {out / 'glyphdraw.pt'}   Samples: {out / 'samples.png'}")
    print("Serve it to the studio:  python pixel_server.py")


if __name__ == "__main__":
    main()
