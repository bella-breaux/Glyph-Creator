#!/usr/bin/env python3
"""
Fine-tune a small model into "glyphgen" with Unsloth (QLoRA), then export it for Ollama.

Works on a free Google Colab T4 (recommended) or a local NVIDIA GPU with 8 GB+ VRAM
(Linux, or Windows via WSL2). See TRAINING.md for the step-by-step guide.

    pip install -r requirements.txt
    python generate_dataset.py
    python train_unsloth.py                          # default base: Gemma 3 1B (fast)
    python train_unsloth.py --base unsloth/gemma-3-4b-it --epochs 1   # better drawings, slower

Output: glyphgen-gguf/*.gguf and glyphgen-gguf/Modelfile, then:
    ollama create glyphgen -f glyphgen-gguf/Modelfile

Library APIs move quickly. This script adapts to the common TRL/Unsloth signature changes,
but if a newer version breaks it, compare with the current Unsloth Gemma/Qwen notebook.
"""
import argparse, glob, inspect, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/gemma-3-1b-it", help="Hugging Face model id (Unsloth 4-bit variants are fastest)")
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--out", default="glyphgen-gguf")
ap.add_argument("--quant", default="q4_k_m", help="GGUF quantization (q4_k_m fits easily in 8 GB)")
a = ap.parse_args()

import torch
from datasets import load_dataset

try:  # Newer Unsloth: one loader for all architectures (Gemma 3 needs it).
    from unsloth import FastModel as Loader
    NEW_API = True
except ImportError:
    from unsloth import FastLanguageModel as Loader
    NEW_API = False

print(f"Loading {a.base} …")
model, tokenizer = Loader.from_pretrained(model_name=a.base, max_seq_length=a.max_len, load_in_4bit=True)
peft_kw = dict(r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=3407)
if NEW_API:
    peft_kw.update(finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True)
else:
    peft_kw.update(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
model = Loader.get_peft_model(model, **peft_kw)

tok = getattr(tokenizer, "tokenizer", tokenizer)  # multimodal processors wrap the text tokenizer
ds = load_dataset("json", data_files={"train": os.path.join(a.data, "train.jsonl"), "eval": os.path.join(a.data, "eval.jsonl")})
def to_text(batch):
    return {"text": [tok.apply_chat_template(m, tokenize=False).removeprefix(tok.bos_token or "\0") for m in batch["messages"]]}
ds = ds.map(to_text, batched=True, remove_columns=["messages"])
sample = ds["train"][0]["text"]
print("Example training text:\n" + sample[:600])

from trl import SFTTrainer, SFTConfig
cfg = dict(dataset_text_field="text", per_device_train_batch_size=a.batch, gradient_accumulation_steps=2, num_train_epochs=a.epochs,
           learning_rate=a.lr, warmup_steps=20, lr_scheduler_type="linear", optim="adamw_8bit", weight_decay=0.01, logging_steps=25,
           eval_strategy="steps", eval_steps=200, save_strategy="no", output_dir="outputs", report_to="none", seed=3407,
           bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported())
cfg_params = inspect.signature(SFTConfig.__init__).parameters
cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = a.max_len
if "eval_strategy" not in cfg_params:
    cfg["evaluation_strategy"] = cfg.pop("eval_strategy")
tr_params = inspect.signature(SFTTrainer.__init__).parameters
tk = {"processing_class" if "processing_class" in tr_params else "tokenizer": tokenizer}
trainer = SFTTrainer(model=model, train_dataset=ds["train"], eval_dataset=ds["eval"], args=SFTConfig(**{k: v for k, v in cfg.items() if k in cfg_params}), **tk)

# Only learn the replies (the recipes), not the prompts.
try:
    from unsloth.chat_templates import train_on_responses_only
    for user_tag, bot_tag in (("<start_of_turn>user\n", "<start_of_turn>model\n"), ("<|im_start|>user\n", "<|im_start|>assistant\n"),
                              ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n")):
        if user_tag in sample and bot_tag in sample:
            trainer = train_on_responses_only(trainer, instruction_part=user_tag, response_part=bot_tag)
            print("Training on responses only.")
            break
except Exception as e:  # not fatal: the model still learns, just a little less efficiently
    print("Skipping responses-only training:", e)

trainer.train()

# Quick check before exporting.
print("\nSample generation:")
msgs = [{"role": "user", "content": "GLYPH: a ghost that bounces with sparkles"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(input_ids=ids, max_new_tokens=200, do_sample=False)[0][ids.shape[1]:], skip_special_tokens=True))

print(f"\nExporting GGUF ({a.quant}) to {a.out}/ …")
os.makedirs(a.out, exist_ok=True)
model.save_pretrained_gguf(a.out, tokenizer, quantization_method=a.quant)
ggufs = sorted(glob.glob(os.path.join(a.out, "**", "*.gguf"), recursive=True) + glob.glob("*.gguf"), key=os.path.getsize, reverse=True)
if not ggufs:
    raise SystemExit("No .gguf file was produced. Check the Unsloth export log above.")
gguf = ggufs[0]
if os.path.dirname(os.path.abspath(gguf)) != os.path.abspath(a.out):
    gguf = shutil.move(gguf, os.path.join(a.out, os.path.basename(gguf)))
stops = {"gemma": "<end_of_turn>", "qwen": "<|im_end|>", "llama": "<|eot_id|>"}
stop = next((v for k, v in stops.items() if k in a.base.lower()), None)
with open(os.path.join(a.out, "Modelfile"), "w") as f:
    f.write(f"FROM ./{os.path.basename(gguf)}\n")
    f.write("PARAMETER temperature 0.3\nPARAMETER num_ctx 2048\n")
    if stop:
        f.write(f'PARAMETER stop "{stop}"\n')
print(f"\nDone. Model: {gguf}\nNext:  ollama create glyphgen -f {a.out}/Modelfile")
