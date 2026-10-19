#!/usr/bin/env python3
"""
Collect openly licensed pixel art (still images and GIFs) for training, with full credit records.

Sources (both are official APIs that report each file's license):
  * Openverse      - search engine for Creative Commons / public-domain media   https://openverse.org
  * Wikimedia Commons - pixel art categories                                    https://commons.wikimedia.org
  * --local        - folders or .zip packs you downloaded yourself (Kenney, OpenGameArt CC0 packs, ...)

Every file gets a row in raw/manifest.jsonl (title, tags, creator, license, source page) and
raw/ATTRIBUTION.md lists the credits that CC BY / BY-SA licenses require.

Usage:
    pip install requests pillow
    python fetch_pixel_art.py                                  # default subject list, 40 files per subject
    python fetch_pixel_art.py --queries cat ghost rocket --per-query 80
    python fetch_pixel_art.py --gifs-only
    python fetch_pixel_art.py --local ~/Downloads/kenney_1-bit-pack.zip --local ~/Downloads/icons/
    python fetch_pixel_art.py --licenses cc0 pdm               # public domain only

Then:  python convert_to_glyphs.py
"""
import argparse, hashlib, io, json, os, re, sys, time, zipfile
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests pillow")

HERE = Path(__file__).resolve().parent
UA = "GlyphStudioDatasetBuilder/1.0 (https://github.com/bella-breaux/Glyph-Creator; personal ML research)"
OPENVERSE = "https://api.openverse.org/v1/images/"
COMMONS = "https://commons.wikimedia.org/w/api.php"
# Licenses that allow reuse and modification. NC/ND are excluded unless you opt in.
DEFAULT_LICENSES = ["cc0", "pdm", "by", "by-sa"]
SUBJECTS = """cat dog ghost skull heart star moon sun cloud rain lightning fire tree flower mushroom apple cherry pizza
cake coffee rocket ufo alien robot car spaceship sword shield potion key coin gem crown bomb ghost pumpkin bat
spider fish bird frog bunny bear panda dragon slime knight wizard pacman invader controller phone music note
headphones smiley face eye hand arrow house castle planet earth snowflake umbrella bell gift balloon trophy""".split()
EXT_OK = {".png", ".gif", ".webp", ".bmp", ".jpg", ".jpeg"}
MAX_BYTES = 6 * 1024 * 1024


class Store:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"
        self.seen_hash, self.seen_src = set(), set()
        if self.manifest.exists():
            for line in self.manifest.open(encoding="utf-8"):
                try:
                    r = json.loads(line); self.seen_hash.add(r["sha256"]); self.seen_src.add(r.get("source_id"))
                except Exception:
                    pass

    def add(self, data, meta):
        h = hashlib.sha256(data).hexdigest()
        if h in self.seen_hash:
            return False
        ext = meta.get("ext") or ".png"
        sub = self.root / meta["source"]; sub.mkdir(exist_ok=True)
        path = sub / f"{h[:16]}{ext}"
        path.write_bytes(data)
        row = {**meta, "sha256": h, "file": str(path.relative_to(self.root)), "bytes": len(data)}
        with self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.seen_hash.add(h); self.seen_src.add(meta.get("source_id"))
        return True


def session():
    s = requests.Session(); s.headers["User-Agent"] = UA; return s


def get(s, url, delay, **kw):
    """GET with polite pacing, retries and Retry-After handling."""
    for attempt in range(4):
        time.sleep(delay)
        try:
            r = s.get(url, timeout=30, **kw)
        except requests.RequestException as e:
            print(f"    network error ({e.__class__.__name__}), retrying"); time.sleep(2 ** attempt); continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "30")); print(f"    rate limited, waiting {wait}s"); time.sleep(wait); continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        return r
    return None


def download(s, url, delay):
    r = get(s, url, delay, stream=True)
    if r is None or r.status_code != 200:
        return None, None
    ctype = r.headers.get("Content-Type", "").split(";")[0].lower()
    if not ctype.startswith("image/"):
        return None, None
    data = io.BytesIO()
    for chunk in r.iter_content(65536):
        data.write(chunk)
        if data.tell() > MAX_BYTES:
            return None, None
    ext = {"image/gif": ".gif", "image/png": ".png", "image/webp": ".webp", "image/jpeg": ".jpg", "image/bmp": ".bmp"}.get(ctype)
    return (data.getvalue(), ext) if ext else (None, None)


def openverse(s, store, queries, per_query, licenses, gifs_only, delay):
    added = 0
    for q in queries:
        got, page = 0, 1
        while got < per_query and page <= 10:
            params = {"q": f"pixel art {q}", "license": ",".join(licenses), "page_size": 20, "page": page, "mature": "false"}
            if gifs_only:
                params["extension"] = "gif"
            r = get(s, OPENVERSE, delay, params=params)
            if r is None or r.status_code != 200:
                print(f"  openverse '{q}': HTTP {getattr(r, 'status_code', 'error')}"); break
            results = r.json().get("results", [])
            if not results:
                break
            for it in results:
                if got >= per_query:
                    break
                sid = "openverse:" + it["id"]
                if sid in store.seen_src:
                    continue
                data, ext = download(s, it["url"], delay)
                if not data:
                    continue
                ok = store.add(data, {
                    "source": "openverse", "source_id": sid, "query": q, "title": it.get("title") or "",
                    "tags": [t["name"] for t in it.get("tags") or [] if t.get("name")][:20],
                    "creator": it.get("creator") or "", "license": it.get("license"), "license_version": it.get("license_version"),
                    "license_url": it.get("license_url"), "page": it.get("foreign_landing_url"), "provider": it.get("provider"), "ext": ext})
                got += ok; added += ok
            page += 1
        print(f"  openverse '{q}': {got} new")
    return added


COMMONS_LICENSE = [(r"^cc0|public domain|^pd", "cc0"), (r"cc[- ]by[- ]sa", "by-sa"), (r"cc[- ]by(?![- ]sa)", "by")]


def commons(s, store, categories, per_cat, licenses, gifs_only, delay):
    added = 0
    for cat in categories:
        got, cont = 0, {}
        while got < per_cat:
            params = {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                      "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", **cont}
            r = get(s, COMMONS, delay, params=params)
            if r is None or r.status_code != 200:
                print(f"  commons '{cat}': HTTP {getattr(r, 'status_code', 'error')}"); break
            j = r.json()
            for pg in (j.get("query", {}).get("pages") or {}).values():
                if got >= per_cat:
                    break
                info = (pg.get("imageinfo") or [{}])[0]; meta = info.get("extmetadata", {})
                if gifs_only and info.get("mime") != "image/gif":
                    continue
                if info.get("size", 0) > MAX_BYTES or not info.get("mime", "").startswith("image/") or info.get("mime") == "image/svg+xml":
                    continue
                lic_raw = (meta.get("LicenseShortName", {}).get("value") or "").lower()
                lic = next((code for pat, code in COMMONS_LICENSE if re.search(pat, lic_raw)), None)
                if lic not in licenses and not (lic == "cc0" and "pdm" in licenses):
                    continue
                sid = "commons:" + str(pg.get("pageid"))
                if sid in store.seen_src:
                    continue
                data, ext = download(s, info["url"], delay)
                if not data:
                    continue
                strip = lambda v: re.sub(r"<[^>]+>", "", v or "").strip()
                ok = store.add(data, {
                    "source": "commons", "source_id": sid, "query": cat, "title": re.sub(r"^File:|\.\w+$", "", pg.get("title", "")),
                    "tags": [c for c in strip(meta.get("Categories", {}).get("value")).split("|") if c][:20],
                    "description": strip(meta.get("ImageDescription", {}).get("value"))[:300],
                    "creator": strip(meta.get("Artist", {}).get("value"))[:200], "license": lic, "license_name": lic_raw,
                    "license_url": meta.get("LicenseUrl", {}).get("value"), "page": info.get("descriptionurl"), "ext": ext})
                got += ok; added += ok
            if "continue" not in j:
                break
            cont = j["continue"]
        print(f"  commons '{cat}': {got} new")
    return added


def local(store, paths, license_):
    """Ingest packs you downloaded by hand. The filename (and folder) becomes the caption."""
    added = 0
    for p in map(Path, paths):
        items = []
        if p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                items = [(n, z.read(n)) for n in z.namelist() if Path(n).suffix.lower() in EXT_OK and not n.startswith("__MACOSX")]
        elif p.is_dir():
            items = [(str(f.relative_to(p)), f.read_bytes()) for f in p.rglob("*") if f.suffix.lower() in EXT_OK]
        elif p.is_file() and p.suffix.lower() in EXT_OK:
            items = [(p.name, p.read_bytes())]
        n = 0
        for name, data in items:
            if len(data) > MAX_BYTES:
                continue
            parts = [x for x in re.split(r"[/\\]", name) if x]
            title = re.sub(r"[_\-]+", " ", Path(parts[-1]).stem)
            tags = [re.sub(r"[_\-]+", " ", x).lower() for x in parts[:-1]][-3:]
            n += store.add(data, {"source": "local", "source_id": f"local:{p.name}:{name}", "query": "", "title": title, "tags": tags,
                                  "creator": "", "license": license_, "page": str(p), "pack": p.name, "ext": Path(name).suffix.lower()})
        print(f"  local '{p}': {n} new"); added += n
    return added


def write_attribution(root):
    rows = [json.loads(l) for l in (root / "manifest.jsonl").open(encoding="utf-8")] if (root / "manifest.jsonl").exists() else []
    need = [r for r in rows if r.get("license") in ("by", "by-sa")]
    with (root / "ATTRIBUTION.md").open("w", encoding="utf-8") as f:
        f.write("# Image credits\n\nFiles under CC BY / CC BY-SA licenses used in this dataset:\n\n")
        for r in need:
            ver = f" {r['license_version']}" if r.get("license_version") else ""
            f.write(f"- \"{r.get('title') or r['file']}\" by {r.get('creator') or 'unknown'}, CC {r['license'].upper()}{ver}, {r.get('page') or ''}\n")
    lic = {}
    for r in rows:
        lic[r.get("license")] = lic.get(r.get("license"), 0) + 1
    return len(rows), lic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "raw"))
    ap.add_argument("--queries", nargs="*", default=SUBJECTS, help="subjects to search for (prefixed with 'pixel art')")
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--commons-cats", nargs="*", default=["Pixel art", "Animated pixel art", "Pixel art icons", "Sprites (computer graphics)"])
    ap.add_argument("--per-cat", type=int, default=300)
    ap.add_argument("--licenses", nargs="*", default=DEFAULT_LICENSES, help="cc0 pdm by by-sa (add by-nc etc. only for private use)")
    ap.add_argument("--gifs-only", action="store_true")
    ap.add_argument("--skip-openverse", action="store_true"); ap.add_argument("--skip-commons", action="store_true")
    ap.add_argument("--local", action="append", default=[], help="folder or .zip you downloaded (repeatable)")
    ap.add_argument("--local-license", default="cc0", help="license to record for --local packs")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (be polite)")
    a = ap.parse_args()
    store, s = Store(a.out), session()
    total = 0
    if a.local:
        print("Local packs:"); total += local(store, a.local, a.local_license)
    if not a.skip_openverse:
        print("Openverse:"); total += openverse(s, store, a.queries, a.per_query, a.licenses, a.gifs_only, a.delay)
    if not a.skip_commons:
        print("Wikimedia Commons:"); total += commons(s, store, a.commons_cats, a.per_cat, a.licenses, a.gifs_only, a.delay)
    n, lic = write_attribution(store.root)
    print(f"\nAdded {total} files this run; {n} in {store.root} by license: {lic}")
    print("Credits for CC BY/BY-SA files: raw/ATTRIBUTION.md.  Next: python convert_to_glyphs.py")


if __name__ == "__main__":
    main()
