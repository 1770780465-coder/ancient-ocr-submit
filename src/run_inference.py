#!/usr/bin/env python3
import json, os, cv2, traceback
from pathlib import Path
from paddleocr import PaddleOCR

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
DET_MODEL_DIR = os.getenv("DET_MODEL_DIR", "/app/models/det")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))

def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(p for p in fallback.rglob("*") if p.suffix.lower() in suffixes)
    return []

def polygon_to_bbox(points, w, h):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0, min(w-1, int(round(min(xs)))))
    y1 = max(0, min(h-1, int(round(min(ys)))))
    x2 = max(0, min(w, int(round(max(xs)))))
    y2 = max(0, min(h, int(round(max(ys)))))
    return [x1, y1, max(0, x2-x1), max(0, y2-y1)]

def infer_one(ocr_engine, img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    raw = ocr_engine.ocr(str(img_path), det=True, rec=True, cls=False)  # 打开识别
    if raw is None or not isinstance(raw, list) or len(raw) == 0:
        return []
    lines = raw[0] if raw[0] is not None else []
    detections = []
    for line in lines:
        if line is None or len(line) < 2:
            continue
        polygon = line[0]
        text_info = line[1]  # (text, confidence)
        text = text_info[0] if isinstance(text_info, (list, tuple)) and len(text_info) > 0 else ""
        score = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
        if not text or score < MIN_SCORE:
            continue
        bbox = polygon_to_bbox(polygon, w, h)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        detections.append({"bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], "text": text})
    detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return detections

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    image_paths = find_images()
    print(f"Found {len(image_paths)} images")

    ocr_engine = PaddleOCR(lang='ch', det_model_dir=DET_MODEL_DIR)
    results = {}
    for idx, img_path in enumerate(image_paths, 1):
        if idx % 50 == 0:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        try:
            results[img_path.stem] = infer_one(ocr_engine, img_path)
        except Exception as e:
            print(f"Error {img_path}: {e}")
            traceback.print_exc()
            results[img_path.stem] = []

    with OUTPUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
