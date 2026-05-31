"""
import_taco.py

Reads TACO annotations, finds images where litter covers at least
MIN_COVERAGE fraction of the image, and copies them into Arbiter's
data/raw/trash/ folder.

Usage:
    python scripts/import_taco.py

Paths assume TACO and Arbiter are siblings under ~/personal/
"""

import json
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
TACO_DIR        = Path.home() / "personal" / "TACO"
ANNOTATIONS     = TACO_DIR / "data" / "annotations.json"
IMAGES_DIR      = TACO_DIR / "data"
OUTPUT_DIR      = Path.home() / "personal" / "Arbiter" / "data" / "raw" / "trash"

# ── Config ─────────────────────────────────────────────────────────────────
MIN_COVERAGE = 0.05   # litter must cover at least 5% of image area to be included

# ───────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    # Build image_id → image metadata lookup
    images = {img["id"]: img for img in coco["images"]}

    # Accumulate total annotation area per image
    coverage = {}  # image_id → total annotation area (pixels)
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        # COCO bbox format: [x, y, width, height]
        ann_area = ann.get("area", 0)
        coverage[img_id] = coverage.get(img_id, 0) + ann_area

    copied = 0
    skipped_coverage = 0
    skipped_missing = 0

    for img_id, total_ann_area in coverage.items():
        img_meta = images[img_id]
        img_w = img_meta["width"]
        img_h = img_meta["height"]
        img_area = img_w * img_h

        if img_area == 0:
            continue

        frac = total_ann_area / img_area

        if frac < MIN_COVERAGE:
            skipped_coverage += 1
            continue

        # TACO stores images in numbered subdirs like data/batch_1/
        file_name = img_meta["file_name"]  # e.g. "batch_1/000001.jpg"
        src = IMAGES_DIR / file_name

        if not src.exists():
            skipped_missing += 1
            continue

        # Rename to avoid collisions with existing TrashNet trash images
        dest_name = f"taco_{img_id:05d}{src.suffix}"
        dest = OUTPUT_DIR / dest_name
        shutil.copy2(src, dest)
        copied += 1

    print(f"Done.")
    print(f"  Copied:           {copied}")
    print(f"  Skipped (low coverage): {skipped_coverage}")
    print(f"  Skipped (missing file): {skipped_missing}")
    print(f"  Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()