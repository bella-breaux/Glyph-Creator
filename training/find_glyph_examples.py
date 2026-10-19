#!/usr/bin/env python3
"""
Find Glyph Matrix designs that people have published on GitHub, so you can build a training set.

What it looks for, in public repositories about Nothing's Glyph Matrix (Phone (3): 25x25, Phone (4a) Pro: 13x13):
  * small image assets: glyph toy previews, frames, sprite sheets  (.png .gif .webp, up to 128 px)
  * frame data in code and JSON: integer arrays with 169 (13x13) or 625 (25x25) values
  * Glyph Studio patterns (format "glyphstudio/1") and GlyphMo-style frame JSON

It only keeps repositories whose license allows reuse (MIT, Apache, BSD, ISC, MPL, Unlicense, CC0 ...).
Repositories with no license are "all rights reserved"; they are skipped unless you pass --include-unlicensed
(then keep the results private and ask the author before sharing anything trained on them).

Output, compatible with convert_to_glyphs.py:
  raw/github/...                 downloaded images        + rows in raw/manifest.jsonl (source "github")
  raw/github_frames.jsonl        frame data found in code/JSON, already resampled to 13x13
  raw/github_repos.json          every repository checked, its license and what was found

Usage:
  set GITHUB_TOKEN=ghp_...        (Windows)   |   export GITHUB_TOKEN=ghp_...   (Linux/macOS)
  python find_glyph_examples.py
  python find_glyph_examples.py --max-repos 150 --include-copyleft
  python find_glyph_examples.py --repos someone/their-glyph-toys anotherone/glyph-art
  python convert_to_glyphs.py        # images -> glyphs/glyphs.jsonl (frame data is merged in automatically)

A token is needed for GitHub code search (any free account: Settings > Developer settings > Personal access
tokens > Fine-grained, "Public repositories (read-only)"). Without one, only repository search and the seed list run.
"""
import argparse, base64, hashlib, io, json, os, re, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests pillow numpy")
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
API = "https://api.github.com"
UA = "GlyphStudioDatasetBuilder/1.0 (+https://github.com/bella-breaux/Glyph-Creator)"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0", "mpl-2.0", "0bsd", "zlib", "cc-by-4.0", "wtfpl", "bsl-1.0"}
COPYLEFT = {"gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", "agpl-3.0", "cc-by-sa-4.0", "epl-2.0"}
# Known Glyph Matrix projects to always check (more are discovered by search).
SEEDS = ["Nothing-Developer-Programme/GlyphMatrix-Developer-Kit", "KenFeng04/GlyphMatrix-Example-Project", "Sergyenko/glyphfun",
         "codester007/glyph-pong", "dfrost90/glyf1", "anamite/GlyphMo"]
REPO_QUERIES = ["glyph matrix", "glyph toy", "glyphmatrix", "nothing phone glyph", "glyph matrix toy", "nothing phone 3 glyph",
                "topic:glyph-matrix", "topic:glyph-toy", "topic:nothing-phone", "topic:nothingphone"]
CODE_QUERIES = ["GlyphMatrixManager", "GlyphMatrixFrame", "setAppMatrixFrame", "DEVICE_25111p", "DEVICE_23112", "com.nothing.glyph.TOY"]
IMG_EXT = {".png", ".gif", ".webp"}
TEXT_EXT = {".kt", ".java", ".json", ".js", ".ts", ".py", ".txt", ".csv"}
SKIP_DIRS = re.compile(r"(^|/)(build|\.gradle|node_modules|\.idea|gradle/wrapper|mipmap-[a-z]+dpi)(/|$)")
GLYPHY = re.compile(r"glyph|toy|matrix|frame|anim|sprite|icon|preview|led|pixel|emoji|face|font", re.I)

N = 13
yy, xx = np.mgrid[0:N, 0:N]
MASK13 = np.hypot(xx - 6, yy - 6) <= 6.5


class GH:
    def __init__(self, token, delay):
        self.s = requests.Session(); self.s.headers.update({"User-Agent": UA, "Accept": "application/vnd.github+json"})
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.delay, self.token = delay, token

    def get(self, path, raw=False, **params):
        url = path if path.startswith("http") else API + path
        for attempt in range(5):
            time.sleep(self.delay)
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if r.status_code in (403, 429) and ("rate limit" in r.text.lower() or r.headers.get("X-RateLimit-Remaining") == "0"):
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60)); wait = max(5, min(900, reset - time.time() + 2))
                print(f"    GitHub rate limit, waiting {int(wait)}s"); time.sleep(wait); continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            if r.status_code != 200:
                return None
            return r.content if raw else r.json()
        return None


def discover(gh, max_repos, extra):
    found = dict.fromkeys(SEEDS + list(extra))
    for q in REPO_QUERIES:
        j = gh.get("/search/repositories", q=q, per_page=50, sort="stars")
        for it in (j or {}).get("items", []):
            found.setdefault(it["full_name"], None)
        print(f"  repo search '{q}': {len((j or {}).get('items', []))} results")
    if gh.token:
        for q in CODE_QUERIES:
            for page in (1, 2):
                j = gh.get("/search/code", q=q, per_page=100, page=page)
                items = (j or {}).get("items", [])
                for it in items:
                    found.setdefault(it["repository"]["full_name"], None)
                print(f"  code search '{q}' p{page}: {len(items)} results")
                if len(items) < 100:
                    break
                time.sleep(6)  # code search allows ~10 requests/minute
    else:
        print("  (no GITHUB_TOKEN: skipping code search, which finds the most projects)")
    names = list(found)[:max_repos]
    print(f"  {len(names)} repositories to check")
    return names


# ---------------------------------------------------------------- frame data in text files
ARRAY = re.compile(r"[\[{(]\s*((?:-?\d+\s*[,;]\s*){168,700}-?\d+)\s*,?\s*[\]})]", re.S)
NAME_BEFORE = re.compile(r"(?:val|var|const|let|int\[\]|private|public|static|final|\"|')\s*([A-Za-z_][\w]*)\s*(?:\"|')?\s*(?::[^=]*)?[=:]\s*(?:intArrayOf|arrayOf|listOf|new\s+int\s*\[\s*\]|\[|\{)?\s*$")


def resample(vals, side):
    a = np.array(vals, float).reshape(side, side)
    if a.max() > 255:                      # some code uses 0-4095 brightness
        a = a / a.max() * 255
    elif 0 < a.max() <= 1:                 # 0/1 masks
        a = a * 255
    if side != N:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((N, N), Image.BOX), float)
    a = np.where(MASK13, np.clip(np.round(a), 0, 255), 0).astype(int)
    return a.flatten().tolist()


def frames_from_text(text):
    """Find 169/625-value integer arrays (one frame each) and Glyph Studio / GlyphMo JSON."""
    out = []
    try:
        j = json.loads(text)
        if isinstance(j, dict) and j.get("format") == "glyphstudio/1":
            return [{"name": j.get("name", ""), "frames": [{"ms": f.get("ms", 100), "px": f["px"]} for f in j.get("frames", []) if len(f.get("px", [])) == 169]}]
        def walk(o, name=""):
            if isinstance(o, list) and len(o) in (169, 625) and all(isinstance(v, (int, float)) for v in o):
                out.append({"name": name, "frames": [{"ms": 100, "px": resample(o, 13 if len(o) == 169 else 25)}]})
            elif isinstance(o, list) and o and all(isinstance(v, list) and len(v) in (13, 25) for v in o) and len(o) == len(o[0]):
                out.append({"name": name, "frames": [{"ms": 100, "px": resample(sum(o, []), len(o))}]})
            elif isinstance(o, list):
                frs = [walk_frame(x) for x in o]
                if o and all(frs):
                    out.append({"name": name, "frames": frs}); return
                for x in o: walk(x, name)
            elif isinstance(o, dict):
                for k, v in o.items(): walk(v, k if isinstance(k, str) else name)
        def walk_frame(x):
            px = x.get("px") or x.get("pixels") or x.get("data") or x.get("frame") if isinstance(x, dict) else x
            if isinstance(px, list) and len(px) in (169, 625) and all(isinstance(v, (int, float)) for v in px):
                ms = x.get("ms") or x.get("duration") or x.get("delay") or 100 if isinstance(x, dict) else 100
                return {"ms": int(ms), "px": resample(px, 13 if len(px) == 169 else 25)}
            return None
        walk(j)
        return out
    except (ValueError, TypeError, KeyError):
        pass
    for m in ARRAY.finditer(text):
        nums = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
        if len(nums) not in (169, 625) or min(nums) < 0 or len(set(nums)) < 2:
            continue
        name = (NAME_BEFORE.search(text[max(0, m.start() - 120):m.start()]) or [None, ""])[1]
        out.append({"name": name, "frames": [{"ms": 100, "px": resample(nums, 13 if len(nums) == 169 else 25)}]})
    return out


def caption_from(path, name, repo_desc):
    stem = re.sub(r"\.\w+$", "", path.split("/")[-1])
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", f"{name} {stem}")
    words = [w.lower() for w in re.split(r"[^A-Za-z]+", words) if len(w) > 1]
    stop = {"ic", "img", "toy", "glyph", "matrix", "frame", "preview", "drawable", "res", "png", "data", "array", "int", "of", "val", "the"}
    seen = []
    for w in words:
        if w not in stop and w not in seen:
            seen.append(w)
    return " ".join(seen[:6]) or (repo_desc or "glyph")[:60]


def scan_repo(gh, full, args, store, frames_out):
    meta = gh.get(f"/repos/{full}")
    if not meta:
        return {"repo": full, "status": "not found"}
    lic = ((meta.get("license") or {}).get("spdx_id") or "none").lower()
    ok = lic in PERMISSIVE or (args.include_copyleft and lic in COPYLEFT) or (args.include_unlicensed and lic in ("none", "noassertion", "other"))
    info = {"repo": full, "license": lic, "stars": meta.get("stargazers_count", 0), "description": meta.get("description") or "", "images": 0, "frames": 0}
    if not ok:
        info["status"] = "skipped: license"; return info
    branch = meta.get("default_branch", "main")
    tree = gh.get(f"/repos/{full}/git/trees/{branch}", recursive=1)
    if not tree:
        info["status"] = "no tree"; return info
    files = [t for t in tree.get("tree", []) if t["type"] == "blob" and not SKIP_DIRS.search(t["path"])]
    imgs = [t for t in files if Path(t["path"]).suffix.lower() in IMG_EXT and t.get("size", 0) < 400_000 and GLYPHY.search(t["path"])]
    texts = [t for t in files if Path(t["path"]).suffix.lower() in TEXT_EXT and 200 < t.get("size", 0) < 1_500_000]
    raw = f"https://raw.githubusercontent.com/{full}/{branch}/"
    for t in imgs[: args.max_files]:
        data = gh.get(raw + t["path"], raw=True)
        if not data:
            continue
        try:
            im = Image.open(io.BytesIO(data)); w, h = im.size
        except Exception:
            continue
        if max(w, h) > 128 and not (w % 13 == 0 and h == 13) and not (w % 25 == 0 and h == 25):
            continue  # big pictures are usually screenshots, not glyphs
        native = 13 if (w, h) == (13, 13) else 25 if (w, h) == (25, 25) else None
        sheet = h in (13, 25) and w > h and w % h == 0
        added = store.add(data, {"source": "github", "source_id": f"github:{full}:{t['path']}", "query": "", "title": caption_from(t["path"], "", info["description"]),
                                 "tags": [full.split("/")[1]], "creator": full.split("/")[0], "license": lic, "page": f"https://github.com/{full}/blob/{branch}/{t['path']}",
                                 "ext": Path(t["path"]).suffix.lower(), "native_matrix": native, "sprite_sheet": h if sheet else None})
        info["images"] += added
    for t in texts[: args.max_files]:
        if Path(t["path"]).suffix.lower() in (".kt", ".java", ".js", ".ts", ".py") and not GLYPHY.search(t["path"]) and t.get("size", 0) > 200_000:
            continue
        data = gh.get(raw + t["path"], raw=True)
        if not data:
            continue
        for g in frames_from_text(data.decode("utf-8", "ignore")):
            if not g["frames"] or not any(any(f["px"]) for f in g["frames"]):
                continue
            row = {"id": hashlib.sha256(json.dumps(g["frames"]).encode()).hexdigest()[:12], "caption": caption_from(t["path"], g["name"], info["description"]),
                   "tags": [full.split("/")[1]], "license": lic, "creator": full.split("/")[0], "page": f"https://github.com/{full}/blob/{branch}/{t['path']}",
                   "frames": g["frames"]}
            if row["id"] not in frames_out:
                frames_out[row["id"]] = row; info["frames"] += 1
    info["status"] = "ok"
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "raw"))
    ap.add_argument("--repos", nargs="*", default=[], help="extra owner/name repositories to check")
    ap.add_argument("--max-repos", type=int, default=80)
    ap.add_argument("--max-files", type=int, default=300, help="per repository, per kind")
    ap.add_argument("--include-copyleft", action="store_true", help="also GPL / CC BY-SA (fine for personal training; check terms before sharing)")
    ap.add_argument("--include-unlicensed", action="store_true", help="also repos with no license (private research only)")
    ap.add_argument("--delay", type=float, default=0.4)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from fetch_pixel_art import Store  # same manifest format as the pixel-art fetcher
    store = Store(a.out)
    gh = GH(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"), a.delay)
    print("Finding repositories:")
    repos = discover(gh, a.max_repos, a.repos)
    frames_path = Path(a.out) / "github_frames.jsonl"
    frames_out = {}
    if frames_path.exists():
        for l in frames_path.open(encoding="utf-8"):
            r = json.loads(l); frames_out[r["id"]] = r
    report = []
    print("Scanning:")
    for i, full in enumerate(repos, 1):
        info = scan_repo(gh, full, a, store, frames_out)
        report.append(info)
        print(f"  [{i}/{len(repos)}] {full}: {info.get('status')} ({info.get('license', '?')}) images +{info.get('images', 0)}, frames +{info.get('frames', 0)}")
    with frames_path.open("w", encoding="utf-8") as f:
        for r in frames_out.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (Path(a.out) / "github_repos.json").write_text(json.dumps(report, indent=1))
    ok = [r for r in report if r.get("status") == "ok"]
    print(f"\nChecked {len(report)} repos ({len(ok)} usable). Images: {sum(r['images'] for r in ok)}, frame sets: {len(frames_out)}.")
    print(f"Skipped for license: {sum(1 for r in report if r.get('status') == 'skipped: license')} (see github_repos.json).")
    print("Next: python convert_to_glyphs.py")


if __name__ == "__main__":
    main()
