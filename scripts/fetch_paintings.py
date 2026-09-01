#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10.3"]
# ///

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

USER_AGENT = "divinci/0.1 (research; CC0 painting corpus)"

USAGE = """Fetch CC0 paintings and rescale them onto equal-area p5 canvases.

    uv run scripts/fetch_paintings.py
    uv run scripts/fetch_paintings.py --count 50
    uv run scripts/fetch_paintings.py --area 400000
"""

CANVAS_AREA = 200_000

FETCH_LONG_EDGE = 1024

KEEP_MEDIUM = re.compile(r"\boil\b|acrylic|\btempera\b|fresco|encaustic", re.I)
DROP_MEDIUM = re.compile(r"gum tempera|palm.?leaf|folio|manuscript|page from|album leaf", re.I)

def is_easel_painting(medium: str) -> bool:
    return bool(KEEP_MEDIUM.search(medium or "")) and not DROP_MEDIUM.search(medium or "")

_last_request = 0.0
MIN_INTERVAL = 0.2

def get(url: str, params: dict | None = None, binary: bool = False, tries: int = 4):
    global _last_request
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "AIC-User-Agent": USER_AGENT}
    )
    for attempt in range(tries):
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            return payload if binary else json.loads(payload)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * 2**attempt)

def as_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None

def canvas_for(width: int, height: int, area: int = CANVAS_AREA) -> tuple[int, int]:
    aspect = width / height
    even = lambda value: max(2, round(value / 2) * 2)
    return even(math.sqrt(area * aspect)), even(math.sqrt(area / aspect))

def to_srgb(image: Image.Image) -> Image.Image:
    profile = image.info.get("icc_profile")
    if profile:
        try:
            return ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except Exception:
            pass
    return image.convert("RGB")

def rescale(data: bytes, area: int):
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return None, "undecodable"

    image = ImageOps.exif_transpose(image) or image
    width, height = image.size
    canvas_w, canvas_h = canvas_for(width, height, area)
    if width < canvas_w or height < canvas_h:
        return None, "smaller than canvas"

    resized = to_srgb(image).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
    return resized, f"{canvas_w}x{canvas_h}"

def cleveland(long_edge: int):
    skip, total = 0, None
    while total is None or skip < total:
        page = get("https://openaccess-api.clevelandart.org/api/artworks/",
                   {"cc0": 1, "has_image": 1, "type": "Painting", "limit": 100, "skip": skip})
        total = page["info"]["total"]
        for row in page["data"]:
            variants = []
            for name, v in (row.get("images") or {}).items():
                if name not in ("web", "print") or not isinstance(v, dict) or not v.get("url"):
                    continue
                size = [as_int(v.get("width")), as_int(v.get("height"))]
                if all(size):
                    variants.append((v["url"], size[0], size[1]))
            if not variants:
                continue
            big = [v for v in variants if max(v[1], v[2]) >= long_edge]
            url, _, _ = min(big or variants, key=lambda v: max(v[1], v[2]))
            creators = row.get("creators") or []
            yield {
                "source": "cleveland",
                "id": str(row["id"]),
                "title": (row.get("title") or "").strip(),
                "artist": (creators[0].get("description") if creators else "") or "",
                "date": str(row.get("creation_date") or ""),
                "medium": str(row.get("technique") or ""),
                "page": str(row.get("url") or ""),
                "image": url,
            }
        skip += 100

ARTIC_RESULT_CAP = 1000

def artic(long_edge: int):
    query = json.dumps({
        "query": {"bool": {"filter": [
            {"term": {"is_public_domain": True}},
            {"term": {"artwork_type_title.keyword": "Painting"}},
            {"exists": {"field": "image_id"}},
        ]}},
        "sort": [{"id": "asc"}],
    }, sort_keys=True)
    fields = "id,title,artist_title,date_end,medium_display,image_id,thumbnail"
    page = 1
    while True:
        payload = get("https://api.artic.edu/api/v1/artworks/search",
                      {"params": query, "fields": fields, "limit": 100, "page": page})
        rows = payload.get("data") or []
        if not rows:
            return
        for row in rows:
            if not row.get("image_id"):
                continue
            yield {
                "source": "artic",
                "id": str(row["id"]),
                "title": (row.get("title") or "").strip(),
                "artist": str(row.get("artist_title") or ""),
                "date": str(row.get("date_end") or ""),
                "medium": str(row.get("medium_display") or ""),
                "page": f"https://www.artic.edu/artworks/{row['id']}",
                "image": f"https://www.artic.edu/iiif/2/{row['image_id']}"
                         f"/full/!{long_edge},{long_edge}/0/default.jpg",
            }
        if page >= (payload.get("pagination") or {}).get("total_pages", 0):
            return
        page += 1
        if page * 100 > ARTIC_RESULT_CAP:
            return

SOURCES = {"cleveland": cleveland, "artic": artic}

def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def pick(pools: dict[str, list[dict]], count: int) -> list[dict]:
    for rows in pools.values():
        rows.sort(key=lambda r: digest(f"{r['source']}:{r['id']}"))
    chosen, depth = [], 0
    while len(chosen) < count:
        drew = False
        for name in sorted(pools):
            if depth < len(pools[name]):
                chosen.append(pools[name][depth])
                drew = True
                if len(chosen) == count:
                    return chosen
        if not drew:
            return chosen
        depth += 1
    return chosen

def split_for(key: str, eval_fraction: float) -> str:
    return "eval" if int(digest(f"split:{key}")[:8], 16) / 0xFFFFFFFF < eval_fraction else "train"

def main() -> None:
    parser = argparse.ArgumentParser(description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1500, help="paintings to download")
    parser.add_argument("--area", type=int, default=CANVAS_AREA,
                        help="pixels per canvas; proportions come from the painting")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES),
                        help="restrict to one museum (repeatable)")
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--refresh", action="store_true", help="re-list the museums")
    parser.add_argument("--all-media", action="store_true",
                        help="keep manuscripts, scrolls and palm-leaf pages too")
    args = parser.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fetch_long_edge = FETCH_LONG_EDGE

    pools: dict[str, list[dict]] = {}
    for name in args.source or sorted(SOURCES):
        cache = args.out / f"index-{name}.jsonl"
        if cache.exists() and not args.refresh:
            pools[name] = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
            print(f"listing {name}: {len(pools[name])} CC0 paintings (cached)")
            continue
        print(f"listing {name} ...", flush=True)
        pools[name] = list(SOURCES[name](fetch_long_edge))
        cache.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                                 for r in pools[name]), encoding="utf-8")
        print(f"  {len(pools[name])} CC0 paintings")

    if not args.all_media:
        for name, rows in pools.items():
            pools[name] = [r for r in rows if is_easel_painting(r.get("medium", ""))]
        print("after medium filter: " + "  ".join(f"{k} {len(v)}" for k, v in sorted(pools.items())))

    candidates = pick(pools, min(args.count * 2, sum(len(p) for p in pools.values())))
    print(f"\ndownloading up to {args.count} of {len(candidates)} candidates "
          f"at {args.area // 1000}k px per canvas ...", flush=True)

    entries, skipped, seen = [], {}, 0
    for row in candidates:
        if len(entries) >= args.count:
            break
        seen += 1
        key = f"{row['source']}:{row['id']}"
        path = images_dir / f"{row['source']}-{row['id']}.png"

        if path.exists():
            with Image.open(path) as existing:
                size = existing.size
            if abs(size[0] * size[1] - args.area) / args.area > 0.05:
                path.unlink()
            else:
                entries.append({**row, "key": key, "canvas": f"{size[0]}x{size[1]}",
                                "width": size[0], "height": size[1], "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "split": split_for(key, args.eval_fraction)})
                continue

        try:
            data = get(row["image"], binary=True)
        except Exception as error:
            skipped[f"download failed ({type(error).__name__})"] = \
                skipped.get(f"download failed ({type(error).__name__})", 0) + 1
            continue

        image, reason = rescale(data, args.area)
        if image is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        image.save(path, format="PNG", optimize=True)
        entries.append({**row, "key": key, "canvas": reason,
                        "width": image.width, "height": image.height, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "split": split_for(key, args.eval_fraction)})
        if len(entries) % 50 == 0:
            print(f"  {len(entries)} kept / {seen} tried", flush=True)

    manifest = args.out / "manifest.jsonl"
    entries.sort(key=lambda e: e["key"])
    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"\n{len(entries)} paintings -> {manifest}")
    for label, counts in (("source", "source"), ("split", "split")):
        tally: dict[str, int] = {}
        for entry in entries:
            tally[entry[counts]] = tally.get(entry[counts], 0) + 1
        print(f"  by {label}: " + "  ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    if skipped:
        print("  skipped: " + "  ".join(f"{k} {v}" for k, v in sorted(skipped.items())))

if __name__ == "__main__":
    main()
