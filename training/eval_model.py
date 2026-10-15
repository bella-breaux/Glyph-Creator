#!/usr/bin/env python3
"""
Score Ollama models on the held-out glyph set, so you can compare a stock model with your trained glyphgen.

    python eval_model.py --models glyphgen gemma4:e2b-it-qat qwen3.5:9b --limit 100
    python eval_model.py --models glyphgen --host https://yourpc.your-tailnet.ts.net:8443

Models whose name contains "glyphgen" get the short "GLYPH:" prompt they were trained on.
Other models get the studio's full recipe guide, read from ../index.html so it stays in sync.

Metrics (higher is better):
  json        reply parses as JSON
  valid       has 1-4 layers with known kinds and motions
  kinds       same set of layer kinds as the reference
  anims       same set of motions as the reference
  values      same emoji / text values as the reference
  exact       kinds, motions and values all match
"""
import argparse, json, os, re, statistics, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def studio_bits():
    html = open(os.path.join(HERE, "..", "index.html"), encoding="utf-8").read()
    arr = lambda n: json.loads(re.search(rf"const {n} = (\[.*?\]);", html).group(1))
    kinds, anims = arr("KINDS"), arr("ANIMS")
    guide = re.search(r"const RECIPE_GUIDE = `(.*?)`;", html, re.S).group(1)
    guide = guide.replace('${KINDS.join(" | ")}', " | ".join(kinds)).replace('${ANIMS.join(" | ")}', " | ".join(anims))
    return kinds, anims, guide


KINDS, ANIMS, GUIDE = studio_bits()
SCHEMA = {"type": "object", "required": ["name", "frames", "ms", "layers"], "properties": {
    "name": {"type": "string"}, "frames": {"type": "integer"}, "ms": {"type": "integer"}, "loop": {"type": "boolean"},
    "layers": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "object", "required": ["kind", "anim"], "properties": {
        "kind": {"type": "string", "enum": KINDS}, "value": {"type": "string"}, "size": {"type": "integer"}, "x": {"type": "integer"},
        "y": {"type": "integer"}, "angle": {"type": "integer"}, "brightness": {"type": "number"}, "anim": {"type": "string", "enum": ANIMS},
        "speed": {"type": "number"}, "start": {"type": "number"}, "end": {"type": "number"}}}}}}


def full_prompt(short):
    """Rebuild the long prompt the studio sends to stock models (mirrors recipePrompt in index.html)."""
    lines = dict(l.split(": ", 1) for l in short.split("\n") if ": " in l)
    ask = lines.get("GLYPH", "")
    if "CURRENT" in lines:
        task = f"Current recipe: {lines['CURRENT']}\nTask: change it as follows: {ask}. Return the whole revised recipe."
    elif "USE_DRAWING" in lines:
        task = f'Task: the user drew a picture. Use a layer with kind "drawing" (size 11) to show it, and animate it: {ask}'
    elif "BEAT_MS" in lines:
        b = int(lines["BEAT_MS"])
        task = (f"Task: this plays while a ringtone rings, one beat every {b} ms. Pick frames and ms so frames*ms equals 1 or 2 beats "
                f"(for example 8 frames of {round(b / 8)} ms per beat), and use beat, pulse or blink with speed equal to the number of beats. Style: {ask}")
    else:
        task = f"Task: {ask}"
    return f"{GUIDE}\n\n{task}\n\nReply with only the JSON recipe."


def ask(host, model, prompt, timeout):
    body = json.dumps({"model": model, "stream": False, "think": False, "format": SCHEMA, "keep_alive": "10m",
                       "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 800},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def sig(rec):
    ls = rec.get("layers", []) if isinstance(rec, dict) else []
    return (sorted(str(l.get("kind")) for l in ls), sorted(str(l.get("anim")) for l in ls),
            sorted(str(l.get("value", "")).upper() for l in ls if l.get("value")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "eval.jsonl"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.limit]
    results = {}
    for model in a.models:
        tuned = "glyphgen" in model.lower()
        score = {k: 0 for k in ("json", "valid", "kinds", "anims", "values", "exact")}; secs = []
        for i, r in enumerate(rows, 1):
            short, ref = r["messages"][0]["content"], json.loads(r["messages"][1]["content"])
            t0 = time.time()
            try:
                raw = ask(a.host, model, short if tuned else full_prompt(short), a.timeout)
            except Exception as e:
                print(f"  [{model}] request failed: {e}"); raw = ""
            secs.append(time.time() - t0)
            try:
                rec = json.loads(raw); score["json"] += 1
            except Exception:
                continue
            ls = rec.get("layers") if isinstance(rec, dict) else None
            if isinstance(ls, list) and 1 <= len(ls) <= 4 and all(l.get("kind") in KINDS and l.get("anim") in ANIMS for l in ls):
                score["valid"] += 1
            (k1, a1, v1), (k2, a2, v2) = sig(rec), sig(ref)
            score["kinds"] += k1 == k2; score["anims"] += a1 == a2; score["values"] += v1 == v2
            score["exact"] += k1 == k2 and a1 == a2 and v1 == v2
            print(f"\r  {model}: {i}/{len(rows)}", end="", flush=True)
        print()
        results[model] = {k: round(100 * v / len(rows), 1) for k, v in score.items()} | {"sec_per_reply": round(statistics.mean(secs), 2)}
    cols = ["json", "valid", "kinds", "anims", "values", "exact", "sec_per_reply"]
    print("\n" + "model".ljust(28) + "".join(c.rjust(14) for c in cols))
    for m, s in results.items():
        print(m.ljust(28) + "".join(f"{s[c]:>13}{'%' if c != 'sec_per_reply' else 's'}" for c in cols))
    out = os.path.join(HERE, "eval_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
