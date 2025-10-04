# save as match_ocr_to_polygons.py
import json
from shapely.geometry import Polygon, Point

# --- Paths: read/write only in OUT_DIR ---
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OUT_DIR = Path(os.getenv("OUT_DIR") or Path(__file__).resolve().parent / "out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLYGONS_PATH   = OUT_DIR / "points_grouped.json"
OCR_BOXES_PATH  = OUT_DIR / "vision_text_boxes.json"
OUTPUT_PATH     = OUT_DIR / "points_with_text.json"

def load_polygons(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        units = json.load(f)
    polygons = []
    for unit in units:
        points = unit.get('points', [])
        # Skip if too few points
        if len(points) < 3:
            continue
        # Ensure closed polygon
        if points[0] != points[-1]:
            points = points + [points[0]]
        if len(points) < 4:
            continue
        try:
            poly = Polygon(points)
            if not poly.is_valid or poly.is_empty:
                continue
            polygons.append({
                "block": unit.get('block'),
                "unit": unit.get('unit'),
                "polygon": poly,
                "polygon_points": points,
            })
        except Exception as e:
            print(f"Skipping unit '{unit.get('unit')}', error: {e}")
    print(f"Loaded {len(polygons)} valid polygons")
    return polygons

def load_ocr_boxes(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        boxes = json.load(f)
    # Accept either "bbox" or "bounds" for Google Vision format
    for box in boxes:
        if "bbox" not in box and "bounds" in box:
            box["bbox"] = box["bounds"]
    return boxes

def match_boxes_to_polygons(polygons, ocr_boxes):
    result = []
    for poly in polygons:
        poly_obj = poly["polygon"]
        block = poly["block"]
        unit = poly["unit"]
        best_texts = []
        best_count = 0
        all_matches = []
        for box in ocr_boxes:
            bbox = box["bbox"]
            text = box.get("text") or box.get("description") or ""
            if not text.strip():
                continue
            # Count how many bbox corners are inside the polygon
            pts_in = sum(poly_obj.contains(Point(x, y)) for (x, y) in bbox)
            if pts_in >= 3:
                all_matches.append(text)
                if pts_in > best_count:
                    best_count = pts_in
                    best_texts = [text]
                elif pts_in == best_count:
                    best_texts.append(text)
        # For this polygon, include all best matches and all matches
        result.append({
            "block": block,
            "unit": unit,
            "polygon_points": poly["polygon_points"],
            "matched_texts": list(sorted(set(best_texts))),
            "all_possible_texts": list(sorted(set(all_matches))),
            "best_text": " ".join(sorted(set(best_texts))),
        })
    return result

if __name__ == "__main__":
    polygons = load_polygons(POLYGONS_PATH)
    ocr_boxes = load_ocr_boxes(OCR_BOXES_PATH)
    matched = match_boxes_to_polygons(polygons, ocr_boxes)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
    print(f"Saved matched polygons with text as {OUTPUT_PATH}")
