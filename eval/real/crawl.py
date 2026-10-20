"""Collect openly licensed photos of analog gauges, with licence and author for every file.

Sources: Wikimedia Commons (category tree under the seed categories) and the Openverse image API.
Only licences that allow redistribution and modification are kept (CC0, public domain, CC BY, CC BY-SA).
Images are fetched as thumbnails up to --width pixels wide. A manifest line is appended per file, so the
crawl can be stopped and resumed. Files reviewed and deleted are listed in rejected.jsonl and never fetched again.

    python eval/real/crawl.py --out "%USERPROFILE%\\Downloads\\gauge_images" --max-gb 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = "GaugeGuardDatasetCrawler/0.1 (https://github.com/yzd767621/gaugeguard)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
OPENVERSE = "https://api.openverse.org/v1/images/"
SEED_CATEGORIES = ["Pressure gauges", "Manometers", "Bourdon tube pressure gauges", "Analog gauges",
                   "Pressure gauges in locomotives", "Steam gauges", "Vacuum gauges", "Thermometers with dial",
                   "Tire pressure gauges", "Gauges (instruments)"]
OPENVERSE_QUERIES = ["pressure gauge", "manometer", "analog gauge", "pressure gauge dial", "psi gauge",
                     "bar gauge dial", "boiler pressure gauge", "dial thermometer", "vacuum gauge",
                     "hydraulic pressure gauge", "air pressure gauge", "steam gauge", "gauge needle"]
OK_LICENSE = re.compile(r"^(cc0|pdm|public domain|cc[- ]by(-sa)?[- ]?\d|cc by(-sa)? \d|by(-sa)?)", re.I)
SKIP_TITLE = re.compile(r"\b(digital|diagram|drawing|schematic|logo|icon|svg|map|chart|graph|blood|sphygmo|"
                        r"railway gauge|track gauge|narrow gauge|gauge railway|wire gauge|rain gauge|watch|wristwatch|"
                        r"casio|citizen|timex|g-shock|chronograph|barometer|barograph|thermograph|hygrometer|clock|"
                        r"altimeter|weather station|wetterstation|postcard|engraving|patent)\b", re.I)


def get_json(url, params, retries=4):
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # rate limits and transient network errors: back off and retry
            if attempt == retries - 1:
                print("  giving up on", full[:120], type(e).__name__, e)
                return None
            time.sleep(5 * 2 ** attempt)


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(3 * 2 ** attempt)
    return None


class Store:
    def __init__(self, out, max_bytes):
        self.out, self.max_bytes = out, max_bytes
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        self.manifest = os.path.join(out, "manifest.jsonl")
        self.seen, self.hashes, self.bytes = set(), set(), 0
        if os.path.exists(self.manifest):
            for line in open(self.manifest, encoding="utf-8"):
                m = json.loads(line)
                self.seen.add(m["source_url"])
                self.hashes.add(m["sha1"])
                self.bytes += m["bytes"]
        rejected = os.path.join(out, "rejected.jsonl")  # reviewed and deleted: never fetch these again
        if os.path.exists(rejected):
            for line in open(rejected, encoding="utf-8"):
                m = json.loads(line)
                self.seen.add(m["source_url"])
                self.hashes.add(m["sha1"])

    def full(self):
        return self.bytes >= self.max_bytes

    def add(self, data, meta):
        sha = hashlib.sha1(data).hexdigest()
        if sha in self.hashes:
            return False
        name = f"{meta['source']}_{sha[:12]}.jpg"
        with open(os.path.join(self.out, "images", name), "wb") as f:
            f.write(data)
        meta.update(file=name, sha1=sha, bytes=len(data))
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.hashes.add(sha)
        self.seen.add(meta["source_url"])
        self.bytes += len(data)
        return True


def commons_files(category, depth, seen_cats):
    """Yield file titles in a category and its subcategories, breadth first, up to depth levels."""
    queue = [(category, 0)]
    while queue:
        cat, d = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont = {}
        while True:
            data = get_json(COMMONS, {"action": "query", "format": "json", "list": "categorymembers",
                                      "cmtitle": f"Category:{cat}", "cmtype": "file|subcat", "cmlimit": 500, **cont})
            if not data:
                break
            for m in data["query"]["categorymembers"]:
                if m["ns"] == 14 and d < depth:
                    queue.append((m["title"].split(":", 1)[1], d + 1))
                elif m["ns"] == 6:
                    yield m["title"]
            if "continue" not in data:
                break
            cont = data["continue"]
            time.sleep(0.5)


def crawl_commons(store, width, depth):
    seen_cats, titles = set(), []
    for cat in SEED_CATEGORIES:
        titles += [t for t in commons_files(cat, depth, seen_cats) if t not in titles]
    titles = [t for t in titles if re.search(r"\.(jpe?g|png|webp)$", t, re.I) and not SKIP_TITLE.search(t)]
    print(f"commons: {len(titles)} candidate files")
    for i in range(0, len(titles), 40):
        if store.full():
            return
        data = get_json(COMMONS, {"action": "query", "format": "json", "titles": "|".join(titles[i:i + 40]),
                                  "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": width,
                                  "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl|ImageDescription"})
        if not data:
            continue
        for p in data["query"]["pages"].values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii.get("width", 0) < 400:
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not OK_LICENSE.match(lic.strip()) or ii["descriptionurl"] in store.seen:
                continue
            img = fetch(ii.get("thumburl") or ii["url"])
            if img and store.add(img, {"source": "commons", "title": p["title"], "source_url": ii["descriptionurl"],
                                       "license": lic, "license_url": meta.get("LicenseUrl", {}).get("value"),
                                       "author": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()}):
                print(f"  + {p['title'][5:70]}  [{lic}]  total {store.bytes / 2**20:.0f} MiB")
            time.sleep(0.3)


def crawl_openverse(store, pages):
    for q in OPENVERSE_QUERIES:
        for page in range(1, pages + 1):
            if store.full():
                return
            data = get_json(OPENVERSE, {"q": q, "license": "cc0,pdm,by,by-sa", "page_size": 20, "page": page,
                                        "mature": "false"})
            if not data or not data.get("results"):
                break
            for r in data["results"]:
                if r["foreign_landing_url"] in store.seen or SKIP_TITLE.search(r.get("title") or ""):
                    continue
                if (r.get("width") or 0) and r["width"] < 400:
                    continue
                img = fetch(r.get("thumbnail") or r["url"])
                if img and store.add(img, {"source": "openverse", "title": r.get("title"),
                                           "source_url": r["foreign_landing_url"], "provider": r.get("provider"),
                                           "license": f"{r['license']} {r.get('license_version') or ''}".strip(),
                                           "license_url": r.get("license_url"), "author": r.get("creator")}):
                    print(f"  + [{q}] {(r.get('title') or '')[:60]}  [{r['license']}]  total {store.bytes / 2**20:.0f} MiB")
                time.sleep(0.3)
            time.sleep(3)  # anonymous Openverse use is rate limited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gb", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--depth", type=int, default=2, help="Commons subcategory depth")
    ap.add_argument("--openverse-pages", type=int, default=10)
    args = ap.parse_args()
    store = Store(args.out, int(args.max_gb * 2**30))
    print(f"resuming with {len(store.hashes)} files, {store.bytes / 2**20:.0f} MiB")
    crawl_commons(store, args.width, args.depth)
    crawl_openverse(store, args.openverse_pages)
    print(f"done: {len(store.hashes)} files, {store.bytes / 2**20:.0f} MiB in {args.out}")


if __name__ == "__main__":
    main()
