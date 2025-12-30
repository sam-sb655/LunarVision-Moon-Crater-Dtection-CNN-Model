import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

# Set paths
TEST_IMAGES_DIR = Path('test')  # Folder with test images
LABELS_DIR = Path('predictions/labels')  # Output directory for predicted labels
MODEL_PATH = 'best.pt'  # Path to your trained model

# Create output directory
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Load model
device = select_device('')
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()

# Inference loop
image_paths = list(TEST_IMAGES_DIR.glob('*.jpg')) + list(TEST_IMAGES_DIR.glob('*.png'))

for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    # Resize and pad
    img_resized = letterbox(img_np, new_shape=(640, 640))[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_resized = np.ascontiguousarray(img_resized)

    # Prepare tensor
    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Run inference
    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Save predictions
    h, w = img_np.shape[:2]
    label_file = LABELS_DIR / f'{img_path.stem}.txt'

    with open(label_file, 'w') as f:
        if pred is not None and len(pred):
            pred[:, :4] = xyxy2xywh(pred[:, :4])  # Convert to YOLO format
            pred[:, [0, 2]] /= w  # x_center, width normalized
            pred[:, [1, 3]] /= h  # y_center, height normalized

            for *box, conf, cls in pred.cpu().numpy():
                cls = int(cls)
                x, y, bw, bh = box
                f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

print(f"\n✅ Predictions saved in: {LABELS_DIR.resolve()}")
