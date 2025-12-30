# 🌕 Lunar Crater Detection - Final Submission

Welcome to the **Lunar Crater Detection** project!  
A deep learning-based system to detect craters from lunar surface images using YOLOv5.  
Built with accuracy, interpretability, and ease of use in mind.

---

## 📁 Project Structure

LunarVision_SDS_BabluBadmosh/
├── best.pt # 🔍 Final trained YOLOv5 model (weights)
├── generate_predictions.py # 🧠 Script to generate YOLO-style predictions on test images
├── test_results(ZIP)/
│ ├── labels/ # 📝 YOLO-format prediction labels (output from inference)
│ └── test_images/ # 🧪 Original test images
├── app.py # 🚀 Gradio-based web UI for crater detection
├── training_notebook.ipynb # 📓 Google Colab notebook for model fine-tuning
├── requirements.txt # 📦 Python dependencies
├── LunarVision_BabluBadmosh_Report # 📒Progress Report
└── README.md # 📘 You are here!

---

## 🧠 Model Details

- **Base Model:** `YOLOv5s` (Ultralytics)
- **Training Dataset:** Custom lunar crater dataset with bounding box annotations
- **Classes:** `crater` (class ID: 0)
- **Final Weights:** `best.pt`, trained for 100 epochs
- **Image Size:** `640x640`
- **Training Environment:** Google Colab with GPU

---

## 📓 Training Instructions

Training is done using `training_notebook.ipynb` in Google Colab.

### 🔧 Steps:

1. Mount Google Drive
2. Clone YOLOv5 repo and install dependencies
3. Organize dataset:
   ```
   /MyDrive/YOLO_Dataset/
   ├── images/train/
   │   └── img1.jpg ...
   └── labels/train/
       └── img1.txt ...
   ```
4. Create `fine_tune.yaml`:

   ```yaml
   path: /content/drive/MyDrive/YOLO_Dataset
   train: images/train
   val: images/train
   nc: 1
   names: ["crater"]
   ```

5. Train YOLOv5 model:

   ```bash
   python train.py \
     --img 640 \
     --batch 16 \
     --epochs 100 \
     --data data/fine_tune.yaml \
     --weights yolov5x.pt \
     --name fine_tune_700
   ```

> 🔁 You can retrain from scratch by running the notebook end-to-end.

---

## 🧪 Inference on Test Set

### Step 1: Add Test Images

Place all `.jpg` or `.png` images into the `test/` folder.

### Step 2: Run the Inference Script

```bash
python generate_predictions.py
Loads best.pt

Runs inference on all test images

Saves YOLO-format predictions in predictions/labels/

Step 3: Create Submission
Zip the predictions/labels/ directory as test_results.zip

Submit the .zip for evaluation

🌐 Web Interface (Gradio)
Launch an interactive GUI for crater detection:

bash
Copy
Edit
python app.py
Features:

Upload any image

View original and annotated image

See individual crater crops

View confidence histogram and crater count

📦 Requirements
Install all dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Key Dependencies:

torch

opencv-python

gradio

seaborn

matplotlib

Pillow

numpy

📦 Submitted Test Results Format
python
Copy
Edit
test_results.zip/
├── labels/         # YOLO-format prediction labels
└── test_images/    # Original test images
✔️ Each .txt in labels/ corresponds to a .jpg in test_images/ (case-sensitive match)

📌 Notes for Evaluator
✅ Fully trained and validated model

✅ Clear structure and usage instructions

✅ Inference output in required format

✅ Interactive demo using Gradio

✅ Setup reproducible via Colab + pip install

👨‍💻 Author Info
Name: Soumya Basuli

Institute: Indian Institute of Technology Dharwad

Submission for: Satellite Data Science (SDS) Project

Team Name: BabluBadmosh 🛰️

🌙 Thank you for evaluating!
```
