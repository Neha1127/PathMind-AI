##centroid
import json
from shapely.geometry import Polygon
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OUT_DIR = Path(os.getenv("OUT_DIR") or Path(__file__).resolve().parent / "out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_JSON  = OUT_DIR / "points_with_text.json"
OUTPUT_JSON = OUT_DIR / "points_with_centroid.json"

def add_centroid_to_polygons(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for unit in data:
        pts = unit["polygon_points"]
        poly = Polygon(pts)
        centroid = poly.centroid
        # Save as [x, y], rounded for readability
        unit["centroid"] = [round(centroid.x, 2), round(centroid.y, 2)]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Done! Wrote centroid info to {output_json}")

if __name__ == "__main__":
    add_centroid_to_polygons(INPUT_JSON, OUTPUT_JSON)
