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

## Ideas to go further

- **Image to glyph:** train on pairs of emoji images and 13×13 frames, or use a vision model as the teacher.
- **Distillation:** generate recipes with a big model (or Claude), keep the best, and train glyphgen on them.
- **Preference tuning (DPO):** use Compare picks as "chosen vs rejected" pairs.

## Troubleshooting

- **Out of memory:** use `--batch 4` (or 2) and `--max-len 768`.
- **Error mentioning `SFTConfig` / `max_seq_length` / `processing_class`:** TRL changed its API. Run `pip install -U unsloth trl`. The script adapts to both old and new names.
- **The model answers in prose instead of JSON:** it trained too little. Use `--epochs 3`, or add your own examples.
- **`ollama create` complains about the template:** add a `TEMPLATE` line for your base model to the `Modelfile`. Copy it from `ollama show --modelfile gemma3:1b`.
