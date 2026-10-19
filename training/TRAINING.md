# Train your own glyph model ("glyphgen")

This turns a small open model into a specialist that writes Glyph Studio **recipes**. A recipe is a short description of emoji, text, shapes and motion that the studio draws precisely. A 1B model trained on this one job can beat general models many times its size, and it runs fast on an 8 GB GPU.

```
description ──▶ glyphgen ──▶ recipe JSON ──▶ studio renderer ──▶ 13×13 frames
"a ghost that bounces"      {"layers":[{"kind":"emoji","value":"👻","anim":"bounce"}]}
```

## 1. Make the training data (any computer, 10 seconds)

```bash
cd training
python generate_dataset.py            # 8,000 training + 300 held-out examples in data/
```

The generator builds varied pairs of descriptions and recipes:

- single subjects (about 80 emoji and 20 shapes × 19 motions × sizes and speeds)
- background effects
- scrolling names
- countdowns
- "A then B" sequences
- heartbeats
- "animate my drawing"
- beat-synced examples (`BEAT_MS`)
- edit requests (`CURRENT:` plus "make it faster", "add sparkles" and so on)

**Add your own taste.** In the studio, when a result looks good (or after you fix its recipe by hand), click **Save as training example**. Later, click **Export examples** and mix them in. Each one counts 5× by default:

```bash
python generate_dataset.py --extra my-glyph-examples.jsonl
```

## 2. Train

### Option A: Google Colab (easiest, free)

1. Go to [colab.research.google.com](https://colab.research.google.com) → **New notebook** → **Runtime → Change runtime type → T4 GPU**.
2. Run these cells:

   ```python
   !git clone https://github.com/bella-breaux/Glyph-Creator.git
   %cd Glyph-Creator/training
   !pip install -q unsloth
   !python generate_dataset.py
   !python train_unsloth.py
   ```

   Training the 1B model takes roughly 20–40 minutes on a T4.
3. Download `glyphgen-gguf/` from the Files panel on the left (the `.gguf` file and `Modelfile`).

### Option B: your PC (8 GB NVIDIA GPU)

Unsloth runs best on Linux. On Windows, use WSL2 (Ubuntu), which can use your NVIDIA GPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python train_unsloth.py
```

### Choosing a base model

| `--base` | VRAM to train | Notes |
|---|---|---|
| `unsloth/gemma-3-1b-it` (default) | ~4 GB | Fast to train and to run. Good for this narrow task. |
| `unsloth/gemma-3-4b-it` | ~7 GB | Better at unusual descriptions. Use `--batch 4` on 8 GB. |
| `unsloth/Qwen2.5-3B-Instruct` | ~6 GB | Good alternative family. |

Newer models such as Gemma 4 and Qwen 3.5 also work if Unsloth supports them. Pass their Hugging Face id with `--base`.

## 3. Install it in Ollama

Copy the `glyphgen-gguf` folder to your PC, then:

```bash
ollama create glyphgen -f glyphgen-gguf/Modelfile
ollama run glyphgen "GLYPH: a cat that bounces"     # quick test: should print a recipe
```

## 4. Use it

Open the studio and pick **glyphgen** in the AI assistant's **Model** list. The studio sees "glyphgen" in the name and sends the short `GLYPH:` prompt it was trained on, so it answers fast. Use **Compare** to put it head to head with a stock model.

## 5. Measure it

```bash
python eval_model.py --models glyphgen gemma4:e2b-it-qat qwen3.5:9b --limit 100
```

This prints, for each model, the share of replies that parse (`json`), are valid recipes (`valid`), and match the reference kinds, motions and emoji/text values, plus seconds per reply. Results are saved to `eval_results.json`.

This before/after comparison makes a solid experiment for a class paper. For example: "Does a fine-tuned 1B model beat a general 9B model on a structured generation task?"

## Collect real pixel art (optional, for a model that draws)

The recipe model above describes glyphs, and the studio draws them. To train a model that **draws pixels itself**, you need real examples. Two scripts build that dataset from openly licensed art:

```bash
pip install requests pillow numpy
python fetch_pixel_art.py                      # Openverse + Wikimedia Commons, ~70 subjects
python fetch_pixel_art.py --gifs-only --per-query 60
python fetch_pixel_art.py --local ~/Downloads/kenney_1-bit-pack.zip   # packs you downloaded yourself
python convert_to_glyphs.py                    # -> glyphs/glyphs.jsonl, glyphs/patterns/, glyphs/contact_sheet.png
```

- **Licenses:** by default the fetcher only keeps CC0, public domain, CC BY and CC BY-SA files. These all allow reuse and changes. It records every file's creator and license in `raw/manifest.jsonl`, and writes the credits that BY licenses require to `raw/ATTRIBUTION.md`. Only add `--licenses by-nc ...` for private, non-commercial experiments.
- **Being polite:** it uses the sites' official APIs, waits 1 s between requests, and backs off when rate limited.
- **Good extra sources to download by hand and pass with `--local`:** Kenney's 1-bit and pixel packs (CC0), OpenGameArt CC0 icon packs, pixelarticons (MIT).
- **The converter** uses the same steps as the studio's "Image or GIF to glyph" panel: background removal, trimming, sharp sampling, and region-based shading. It skips photos and empty or solid blobs. Open `contact_sheet.png` to spot bad conversions, and any file in `glyphs/patterns/` opens in the studio with **Import…**.

## Find published Glyph Matrix designs (GitHub)

`find_glyph_examples.py` searches GitHub for Glyph Matrix projects (Phone (3) 25×25 and Phone (4a) Pro 13×13) and collects what they publish:

- glyph images: toy previews, frames and sprite sheets
- frame arrays in Kotlin, Java and JSON
- Glyph Studio patterns

It checks a list of known projects plus everything that GitHub search turns up. 25×25 designs are resampled to 13×13.

```bash
set GITHUB_TOKEN=ghp_your_token          # Windows (on Linux/macOS: export GITHUB_TOKEN=...)
python find_glyph_examples.py
python convert_to_glyphs.py              # merges everything into glyphs/glyphs.jsonl
```

- **Token:** a free GitHub token (Settings → Developer settings → Fine-grained token, public repositories read-only) turns on code search, which finds far more projects.
- **Licenses:** only repositories with a reuse-friendly license are kept (MIT, Apache, BSD, CC0, Unlicense and similar). `raw/github_repos.json` shows every repo checked and why any were skipped.
  - `--include-copyleft` also keeps GPL and CC BY-SA projects.
  - `--include-unlicensed` keeps projects with no license. That's "all rights reserved", so keep it private and ask the author first.
- **Designs that aren't on GitHub:** Nothing's own built-in toys and the Nothing Community gallery aren't published as files with reuse terms, so the script doesn't scrape them. If you like a creator's design, ask them, and add it with `fetch_pixel_art.py --local`.

## Train the pixel model (glyphdraw)

A small text-to-glyph model trained **from scratch**. It draws the 137 LEDs one at a time in 5 brightness levels and uses a pre-trained sentence encoder to understand your description. It has about 5 M parameters.

```bash
pip install torch numpy pillow sentence-transformers
python synth_glyphs.py                   # 20k starter glyphs: shapes, arrows, letters, faces, emoji
python train_pixel_model.py              # -> pixel-model/glyphdraw.pt and samples.png
python pixel_server.py                   # serves it on http://127.0.0.1:11501
```

- **Hardware:** it trains in about 20–60 minutes on your 24 GB GPU, and also works on the 8 GB card. On Windows, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).
- **Data:** real glyphs (from the fetch, find and convert steps) are counted 4× by default, so they shape the style while the synthetic set teaches the basics. For a bigger model with lots of real data, use `--d 384 --layers 8 --epochs 40`.
- **Checking progress:** open `pixel-model/samples.png` every few epochs. It shows 4 drawings each for a handful of test prompts.
- **In the studio:** the setup script already publishes port 8444, so once `pixel_server.py` is running, **Pixel model (glyphdraw)** appears in the Model list. It draws 4 options, and you tap the one you want. Then pick a recipe model and **Animate the current frame** to bring it to life. It also works in **Compare**.

## Start from scratch, or fine-tune Gemma/Qwen?

It depends on what the model outputs:

| Model | Output | How to train |
|---|---|---|
| **glyphgen** (text → recipe) | Short JSON text | **Fine-tune** an existing model (Gemma, Qwen) with LoRA. It already understands English and JSON; you only teach it the recipe format and your style. From scratch would need billions of words of text. |
| **A pixel drawer** (text → 13×13 frames) | 169 numbers per frame | **Small model from scratch**, with a **pre-trained text encoder** (for example a sentence-transformer or CLIP) that turns the description into numbers. At 13×13 the drawer itself is tiny (about 1–10 M parameters), so it trains in an hour or two on a 24 GB GPU using `glyphs.jsonl`. |

Fine-tuning a language model to write 169 digits per frame works poorly. It's the same task stock models already struggle with in Pixels mode, so use the two-model split above. The drawer's output can be shown through a `drawing` layer, so recipes can still animate it.

## Ideas to go further

- **Distillation:** generate recipes with a big model (or Claude), keep the best, and train glyphgen on them.
- **Preference tuning (DPO):** use Compare picks as "chosen vs rejected" pairs.

## Troubleshooting

- **Out of memory:** use `--batch 4` (or 2) and `--max-len 768`.
- **Error mentioning `SFTConfig` / `max_seq_length` / `processing_class`:** TRL changed its API. Run `pip install -U unsloth trl`. The script adapts to both old and new names.
- **The model answers in prose instead of JSON:** it trained too little. Use `--epochs 3`, or add your own examples.
- **`ollama create` complains about the template:** add a `TEMPLATE` line for your base model to the `Modelfile`. Copy it from `ollama show --modelfile gemma3:1b`.
