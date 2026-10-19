#!/usr/bin/env python3
"""
Serve your trained pixel model (glyphdraw) to Glyph Studio.

    python pixel_server.py                       # http://127.0.0.1:11501
    python pixel_server.py --model pixel-model/glyphdraw.pt --port 11501

The setup script already publishes this port on your tailnet as https://<pc>.<tailnet>.ts.net:8444,
and the studio looks for it there automatically (or at http://localhost:11501 on the PC itself).
In the studio, pick "Pixel model (glyphdraw)" in the Model list.

Endpoints:
  GET  /health   -> {"ok": true, "model": "...", "val_loss": ...}
  POST /draw     {"prompt": "a ghost", "n": 4, "temperature": 0.9, "guidance": 3}
                 -> {"candidates": [[169 brightness values], ...], "ms": 840}
"""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import torch

import pixel_model as pm

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "pixel-model" / "glyphdraw.pt"))
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1; Tailscale serve publishes it privately")
    ap.add_argument("--port", type=int, default=11501)
    ap.add_argument("--origins", default="*", help="comma-separated origins allowed to call the server (CORS)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if not Path(a.model).exists():
        raise SystemExit(f"No model at {a.model}. Train one first: python train_pixel_model.py")
    dev = torch.device(a.device)
    model, enc, ck = pm.load(a.model, dev)
    lock = Lock()
    allowed = [o.strip() for o in a.origins.split(",")]
    info = {"ok": True, "model": "glyphdraw", "encoder": ck["encoder"], "val_loss": ck["meta"].get("val_loss"),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    print(f"Loaded {a.model} ({info['params_m']} M params, {ck['encoder']} encoder) on {dev}")

    class H(BaseHTTPRequestHandler):
        def cors(self):
            o = self.headers.get("Origin", "")
            self.send_header("Access-Control-Allow-Origin", "*" if "*" in allowed else (o if o in allowed else allowed[0]))
            self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def reply(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.cors(); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)

        def do_OPTIONS(self):
            self.send_response(204); self.cors(); self.end_headers()

        def do_GET(self):
            self.reply(200, info) if self.path.rstrip("/") in ("", "/health") else self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/draw":
                return self.reply(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 10000)) or b"{}")
                prompt = str(body.get("prompt", "")).strip()[:200]
                if not prompt:
                    return self.reply(400, {"error": "prompt is empty"})
                n = max(1, min(8, int(body.get("n", 4))))
                temp = max(0.1, min(2.0, float(body.get("temperature", 0.9))))
                guide = max(0.0, min(10.0, float(body.get("guidance", 3.0))))
                t0 = time.time()
                with lock:
                    t = torch.tensor(enc.encode([prompt])[0], device=dev)
                    toks = model.sample(t, n=n, temperature=temp, guidance=guide).cpu().numpy()
                self.reply(200, {"candidates": [pm.tokens_to_px(tk) for tk in toks], "ms": int((time.time() - t0) * 1000)})
            except Exception as e:
                self.reply(500, {"error": str(e)[:200]})

        def log_message(self, fmt, *args):
            pass

    print(f"Pixel model ready on http://{a.host}:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
