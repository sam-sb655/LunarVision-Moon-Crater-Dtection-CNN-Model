import pathlib
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import torch
import datetime
import matplotlib.pyplot as plt

pathlib.PosixPath = pathlib.WindowsPath
sys.path.insert(0, os.path.join(os.getcwd(), 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
model.eval()

def detect_craters(image, brightness=1.0, contrast=1.0):
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_np = cv2.convertScaleAbs(img_np, alpha=contrast, beta=(brightness - 1.0) * 255)

    img0 = img_np.copy()
    img = letterbox(img_np, new_shape=(640, 640))[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/detect/gradio_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    single_craters = []
    confidences = []

    if pred is not None and len(pred):
        pred = pred.cpu().numpy()
        for i, (*box, conf, cls) in enumerate(pred):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img0, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            crop = img_np[y1:y2, x1:x2]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                single_craters.append(Image.fromarray(crop_rgb))
                confidences.append(conf)

    result_img_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(result_img_path, img0)

    fig, ax = plt.subplots()
    ax.hist([float(c) * 100 for c in confidences], bins=10, color='cornflowerblue')
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Crater Count")
    plt.tight_layout()

    original_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(original_rgb), Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)), single_craters, fig, len(single_craters)

with gr.Blocks(title="Lunar Crater Detector") as demo:
    gr.Markdown("## 🌕 Lunar Crater Detection (BabluBadmosh)")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Moon Surface Image", scale=4)
        run_button = gr.Button("🚀 Detect Craters", scale=1)

    with gr.Row():
        brightness = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Brightness")
        contrast = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Contrast")

    with gr.Row():
        original = gr.Image(label="🟡 Original Image")
        annotated = gr.Image(label="🟢 Craters Detected")

    crater_gallery = gr.Gallery(label="🔍 Individual Craters", columns=3, rows=2)
    confidence_plot = gr.Plot(label="📊 Confidence Histogram")
    crater_count_display = gr.Number(label="🧮 Total Craters")

    run_button.click(
        fn=detect_craters,
        inputs=[input_image, brightness, contrast],
        outputs=[original, annotated, crater_gallery, confidence_plot, crater_count_display]
    )

demo.launch()
