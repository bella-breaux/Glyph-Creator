#!/usr/bin/env python3
"""
One command: collect glyph data, train the pixel model (glyphdraw), and start serving it to the studio.

    python auto_train.py                    # everything, then serves the model
    python auto_train.py --skip-fetch       # train on synthetic + whatever is already downloaded
    python auto_train.py --big              # larger model for a 24 GB GPU (slower, better with lots of data)
    python auto_train.py --no-serve         # stop after training

Steps (each one is skipped if its output already exists, unless you pass --fresh):
  1. synthetic starter glyphs                     synth_glyphs.py
  2. Glyph Matrix designs on GitHub               find_glyph_examples.py   (needs GITHUB_TOKEN for code search)
  3. openly licensed pixel art and GIFs           fetch_pixel_art.py
  4. convert everything to 13x13 glyphs           convert_to_glyphs.py
  5. train glyphdraw                              train_pixel_model.py
  6. serve it on http://127.0.0.1:11501            pixel_server.py

A log of every step goes to training/auto_train.log. Safe to stop and re-run: finished steps are skipped.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "auto_train.log"


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, step):
    say(f"--- {step}: {' '.join(args)}")
    t0 = time.time()
    with LOG.open("a", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, *args], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print("    " + line.rstrip(), flush=True); log.write("    " + line)
        p.wait()
    ok = p.returncode == 0
    say(f"{'done' if ok else 'FAILED'}: {step} ({time.time() - t0:.0f}s)")
    return ok


def count_lines(p):
    try:
        return sum(1 for _ in open(p, encoding="utf-8"))
    except OSError:
        return 0


def check_torch():
    try:
        import torch
    except ImportError:
        say("PyTorch is not installed. Run setup/train-windows.ps1 (Windows) or: pip install torch numpy pillow sentence-transformers requests")
        return None
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0); mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        say(f"GPU: {name} ({mem:.0f} GB)"); return mem
    say("No CUDA GPU visible to PyTorch: training will run on the CPU (much slower). If you have an NVIDIA card, reinstall torch with CUDA (setup/train-windows.ps1).")
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-fetch", action="store_true", help="don't download anything new")
    ap.add_argument("--fresh", action="store_true", help="redo every step even if its output exists")
    ap.add_argument("--big", action="store_true", help="bigger model (d=384, 8 layers, 40 epochs)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--synthetic", type=int, default=20000)
    ap.add_argument("--per-query", type=int, default=40, help="pixel-art downloads per subject")
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "hash"])
    ap.add_argument("--no-serve", action="store_true")
    a = ap.parse_args()
    LOG.write_text("")
    say("Glyph Studio auto-train starting")
    gpu = check_torch()
    if gpu is None:
        sys.exit(1)
    glyphs, raw = HERE / "glyphs", HERE / "raw"

    if a.fresh or count_lines(glyphs / "synthetic.jsonl") < a.synthetic:
        run(["synth_glyphs.py", "--n", str(a.synthetic)], "1/6 synthetic glyphs")
    else:
        say("1/6 synthetic glyphs: already there")

    if not a.skip_fetch:
        if os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
            run(["find_glyph_examples.py"], "2/6 Glyph Matrix designs on GitHub")
        else:
            say("2/6 GitHub designs: running without GITHUB_TOKEN (repository search only; set a token to find far more)")
            run(["find_glyph_examples.py"], "2/6 Glyph Matrix designs on GitHub")
        run(["fetch_pixel_art.py", "--per-query", str(a.per_query)], "3/6 openly licensed pixel art")
    else:
        say("2-3/6 downloads: skipped (--skip-fetch)")

    if (raw / "manifest.jsonl").exists() or (raw / "github_frames.jsonl").exists():
        run(["convert_to_glyphs.py"], "4/6 convert to 13x13 glyphs")
    else:
        say("4/6 convert: nothing downloaded, training on synthetic glyphs only")
    real = count_lines(glyphs / "glyphs.jsonl")
    say(f"Data: {count_lines(glyphs / 'synthetic.jsonl')} synthetic + {real} real glyphs")

    model = HERE / "pixel-model" / "glyphdraw.pt"
    if a.fresh or not model.exists():
        args = ["train_pixel_model.py", "--encoder", a.encoder]
        if a.big:
            args += ["--d", "384", "--layers", "8", "--epochs", str(a.epochs or 40)]
        elif a.epochs:
            args += ["--epochs", str(a.epochs)]
        if gpu and gpu < 10:
            args += ["--batch", "128"]
        if not run(args, "5/6 train glyphdraw"):
            say("Training failed; see the lines above and auto_train.log"); sys.exit(1)
    else:
        say("5/6 train: a model already exists (pass --fresh to retrain)")
    say(f"Samples to check: {HERE / 'pixel-model' / 'samples.png'}")

    if a.no_serve:
        say("Done. Start the server any time with: python pixel_server.py"); return
    say("6/6 serving. In the studio pick \"Pixel model (glyphdraw)\". Leave this window open; Ctrl+C stops it.")
    run(["pixel_server.py"], "6/6 pixel server")


if __name__ == "__main__":
    main()
