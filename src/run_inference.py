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

# 加载字符映射
with open(CHAR_DICT_PATH, 'r', encoding='utf-8') as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

# 加载自定义分类器
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_crop(crop_bgr):
    try:
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            pred = out.argmax(dim=1).item()
        return idx_to_char[pred]
    except Exception:
        return "?"  # 确保不返回空字符串

def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if INPUT_DIR.exists():
        return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in suffixes)
    fallback = Path("/saisdata")
    if fallback.exists():
        return sorted(p for p in fallback.rglob("*") if p.suffix.lower() in suffixes)
    return []

def polygon_to_bbox(polygon, w, h):
    """将 polygon 转换为 [x, y, w, h]，兼容平坦坐标和点列表格式"""
    if not isinstance(polygon, (list, np.ndarray)) or len(polygon) == 0:
        return [0, 0, 0, 0]
    # 平坦坐标 [x1, y1, x2, y2]
    if isinstance(polygon[0], (int, float, np.number)):
        if len(polygon) >= 4:
            x1, y1, x2, y2 = map(float, polygon[:4])
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
        else:
            return [0, 0, 0, 0]
    else:
        # 点列表 [[x1,y1], ...]
        try:
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        except (IndexError, TypeError):
            return [0, 0, 0, 0]
    x1 = max(0, min(w-1, int(round(x_min))))
    y1 = max(0, min(h-1, int(round(y_min))))
    x2 = max(0, min(w, int(round(x_max))))
    y2 = max(0, min(h, int(round(y_max))))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return [0, 0, 0, 0]
    return [x1, y1, bw, bh]

def infer_one(ocr_engine, img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    # 关键修改：使用 rec=True 避免 PaddleOCR 内部 dt_boxes 判断 bug
    raw = ocr_engine.ocr(str(img_path), det=True, rec=True, cls=False)
    if raw is None or not isinstance(raw, list) or len(raw) == 0:
        return []
    lines = raw[0] if raw[0] is not None else []
    detections = []
    for line in lines:
        if line is None or len(line) < 2:
            continue
        polygon = line[0]
        # PaddleOCR 自带识别结果（忽略，仅用其检测框）
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
