# ocr_extract.py  (no CLI, env-only paths)
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv        # pip install python-dotenv
from google.cloud import vision                    # pip install google-cloud-vision

# Load .env once (searches upward)
load_dotenv(find_dotenv())

# Folder where this script lives
BASE = Path(__file__).resolve().parent

# 🔧 Take image from .env; fallback to the old local file
DEFAULT_IMAGE = Path(os.getenv("IMAGE_PATH") or (BASE / "First Floor Plan_page-0001.jpg"))

# 🔧 Always write to OUT_DIR/vision_text_boxes.json (fallback to local if OUT_DIR missing)
DEFAULT_OUT = (Path(os.getenv("OUT_DIR")) / "vision_text_boxes.json") if os.getenv("OUT_DIR") else (BASE / "vision_text_boxes.json")
DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)


def pick_image() -> Path:
    """Always use the DEFAULT_IMAGE. Exit if it doesn't exist."""
    if DEFAULT_IMAGE.exists():
        return DEFAULT_IMAGE
    sys.exit(
        f"Image not found:\n  {DEFAULT_IMAGE}\n"
        "Set IMAGE_PATH in your .env or place the file at the fallback path."
    )


def detect_text_with_bounds(image_path: Path, output_json: Path) -> int:
    # Ensure Vision credentials are configured and the file exists
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        sys.exit(
            "GOOGLE_APPLICATION_CREDENTIALS not set or file missing.\n"
            "Add it to your .env pointing to the service-account JSON."
        )
    # Make sure the Vision client picks it up even if parent shell didn't export it
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    
    client = vision.ImageAnnotatorClient()
    with image_path.open("rb") as f:
        image = vision.Image(content=f.read())

    resp = client.text_detection(image=image)
    if resp.error.message:
        sys.exit(f"Vision API error: {resp.error.message}")

    texts = resp.text_annotations[1:]  # index 0 = full text
    boxes = [
        {"text": t.description,
         "bbox": [(v.x, v.y) for v in t.bounding_poly.vertices]}
        for t in texts
    ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved {len(boxes)} boxes → {output_json}")
    print(f"[i] Image: {image_path}")
    return len(boxes)

def main():
    img = pick_image()
    out = Path(os.getenv("OCR_OUT", DEFAULT_OUT))
    detect_text_with_bounds(img, out)

if __name__ == "__main__":
    main()
