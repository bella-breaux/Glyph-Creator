"""
Shared pieces of the Glyph Studio pixel model ("glyphdraw"): the 13x13 circle, the text encoders,
the network, and sampling. Used by train_pixel_model.py and pixel_server.py.

The model draws a glyph one LED at a time (137 LEDs, raster order), like a tiny image-GPT:
  [text embedding] -> LED 1 -> LED 2 -> ... -> LED 137
Each LED is one of LEVELS brightness steps. Classifier-free guidance makes it follow the prompt more closely.
"""
import hashlib, math, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 13
_yy, _xx = np.mgrid[0:N, 0:N]
MASK = (np.hypot(_xx - 6, _yy - 6) <= 6.5).flatten()
IDX = np.nonzero(MASK)[0]            # the 137 LED positions in the 169 grid
assert len(IDX) == 137
LEVELS = 5                           # 0 = off, 1..4 = brightness steps
LEVEL_PX = [0, 64, 128, 192, 255]


def px_to_tokens(px):
    a = np.asarray(px, float)[IDX]
    return np.clip(np.round(a / 255 * (LEVELS - 1)), 0, LEVELS - 1).astype(np.int64)


def tokens_to_px(tok):
    out = np.zeros(N * N, int); out[IDX] = [LEVEL_PX[int(t)] for t in tok]
    return out.tolist()


# ---------------------------------------------------------------- text encoders
class HashEncoder:
    """No-download fallback: hashed words + word pairs + character trigrams. Fine for a small vocabulary."""
    name, dim = "hash", 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            feats = words + [a + "_" + b for a, b in zip(words, words[1:])] + [w[j:j + 3] for w in words for j in range(max(1, len(w) - 2))]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
            n = np.linalg.norm(out[i]); out[i] /= n if n else 1
        return out


class MiniLMEncoder:
    """Sentence-transformers all-MiniLM-L6-v2 (downloads ~90 MB once). Understands synonyms and new phrasing."""
    name, dim = "minilm", 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.m.encode(list(texts), batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def make_encoder(name, strict=False):
    if name == "minilm":
        try:
            return MiniLMEncoder()
        except Exception as e:  # not installed or offline
            if strict:  # a model trained on MiniLM can't use another encoder
                raise SystemExit(f"This model needs sentence-transformers: pip install sentence-transformers  ({e})")
            print(f"MiniLM unavailable ({e.__class__.__name__}: {e}); using the hash encoder instead.")
    return HashEncoder()


# ---------------------------------------------------------------- network
class GlyphDraw(nn.Module):
    def __init__(self, text_dim, d=256, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.cfg = dict(text_dim=text_dim, d=d, layers=layers, heads=heads, dropout=dropout)
        self.cond = nn.Sequential(nn.Linear(text_dim, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(d))                 # "no prompt", for classifier-free guidance
        self.tok = nn.Embedding(LEVELS, d)
        self.pos = nn.Parameter(torch.randn(len(IDX) + 1, d) * 0.02)
        # 2-D position of each LED, so the model knows which LEDs are neighbours.
        yx = np.stack([IDX // N, IDX % N], 1) / (N - 1) * 2 - 1
        self.register_buffer("yx", torch.tensor(np.concatenate([[[0, 0]], yx]), dtype=torch.float32))
        self.pos2d = nn.Linear(2, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, LEVELS)
        L = len(IDX) + 1
        self.register_buffer("causal", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, text, tokens, drop_mask=None):
        """text: (B, text_dim); tokens: (B, T<=137) previous LEDs. Returns logits (B, T+1, LEVELS)."""
        c = self.cond(text)
        if drop_mask is not None:
            c = torch.where(drop_mask[:, None], self.null.expand_as(c), c)
        x = torch.cat([c[:, None], self.tok(tokens)], 1)
        T = x.shape[1]
        x = x + self.pos[:T] + self.pos2d(self.yx[:T])
        x = self.body(x, mask=self.causal[:T, :T])
        return self.head(self.norm(x))

    @torch.no_grad()
    def sample(self, text, n=4, temperature=0.9, guidance=3.0, top_k=0):
        """text: (text_dim,) tensor. Returns n samples of 137 tokens (LongTensor (n, 137))."""
        self.eval()
        t = text[None].expand(n, -1)
        toks = torch.zeros(n, 0, dtype=torch.long, device=text.device)
        for _ in range(len(IDX)):
            both = self.forward(torch.cat([t, t]), torch.cat([toks, toks]),
                                drop_mask=torch.cat([torch.zeros(n, dtype=torch.bool, device=text.device), torch.ones(n, dtype=torch.bool, device=text.device)]))
            cond, unc = both[:n, -1], both[n:, -1]
            logits = (unc + guidance * (cond - unc)) / max(1e-3, temperature)
            if top_k:
                v, _ = logits.topk(top_k, -1); logits[logits < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            toks = torch.cat([toks, nxt], 1)
        return toks


def save(path, model, encoder_name, meta):
    torch.save({"state": model.state_dict(), "cfg": model.cfg, "encoder": encoder_name, "levels": LEVELS, "meta": meta}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GlyphDraw(**ck["cfg"]).to(device); m.load_state_dict(ck["state"]); m.eval()
    return m, make_encoder(ck["encoder"], strict=True), ck
