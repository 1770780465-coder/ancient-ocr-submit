#!/usr/bin/env python3
import json, os, cv2, torch, traceback
from pathlib import Path
from paddleocr import PaddleOCR
from torchvision import transforms, models
import numpy as np

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_recognition_model.pth")
CHAR_DICT_PATH = os.getenv("CHAR_DICT_PATH", "/app/models/char_dict.json")
DET_MODEL_DIR = os.getenv("DET_MODEL_DIR", "/app/models/det")
IMG_SIZE = 64
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))

with open(CHAR_DICT_PATH, 'r', encoding='utf-8') as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, len(char_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def classify_crop(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(dim=1).item()
    return idx_to_char[pred]

def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(p for p in fallback.rglob("*") if p.suffix.lower() in suffixes)
    return []

def normalize_ocr_lines(result):
    if result is None:
        return []
    try:
        if isinstance(result, list) and len(result) > 0:
            return [list(line) for line in result[0] if line is not None and len(line) >= 2]
    except Exception:
        pass
    return []

def polygon_to_bbox(polygon, w, h):
    """接收统一格式的 polygon（四点嵌套列表）并转换为 [x, y, w, h]"""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x1 = max(0, min(w-1, int(round(min(xs)))))
    y1 = max(0, min(h-1, int(round(min(ys)))))
    x2 = max(0, min(w, int(round(max(xs)))))
    y2 = max(0, min(h, int(round(max(ys)))))
    return [x1, y1, max(0, x2-x1), max(0, y2-y1)]

def infer_one(ocr_engine, img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    raw = ocr_engine.ocr(str(img_path), det=True, rec=False, cls=False)
    lines = normalize_ocr_lines(raw)
    detections = []
    for line in lines:
        if len(line) < 2:
            continue
        polygon = line[0]          # 可能是嵌套列表，也可能是平坦列表
        score = float(line[1][1]) if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
        if score < MIN_SCORE:
            continue

        # ---- 统一坐标格式为四点嵌套列表 ----
        if isinstance(polygon, (list, np.ndarray)) and len(polygon) > 0:
            # 如果第一个元素是数字，说明是平坦列表 [x1, y1, x2, y2] 或类似
            if isinstance(polygon[0], (int, float, np.number)):
                if len(polygon) == 4:
                    x1, y1, x2, y2 = map(float, polygon)
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    continue   # 无法识别的格式，跳过
            # 否则保持原样（已经是点列表）
        else:
            continue

        bbox = polygon_to_bbox(polygon, w, h)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        x, y, bw, bh = bbox
        crop = img[y:y+bh, x:x+bw]
        if crop.size == 0:
            continue
        text = classify_crop(crop)
        detections.append({"bbox": [int(x), int(y), int(bw), int(bh)], "text": text})
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
