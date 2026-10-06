# SignalGuard — Usage Guide

## 1. Installation

### Prerequisites
- Python 3.9 or later
- pip (latest)
- Git

### Steps

```bash
git clone https://github.com/your-username/sigdetect.git
cd sigdetect
pip install -r requirements.txt
```

Verify your environment:

```bash
python src/check_env.py
```

### Optional dependencies

| Package     | Purpose                        | Install command                |
|-------------|--------------------------------|--------------------------------|
| paddleocr   | License-plate OCR              | `pip install paddleocr`        |
| streamlit   | Interactive dashboard          | `pip install streamlit`        |
| pytest      | Running test suite             | `pip install pytest`           |

> **Note:** The pipeline runs without `paddleocr` (OCR is simply skipped), but
> plate compliance checking will not work.

---

## 2. Collecting More Data

1. Place raw images (JPEG/PNG) in `data/raw/images/`.
2. Place matching annotations (Pascal VOC XML) in `data/raw/annotations/`.
3. Re-run the data preparation script:

```bash
python src/prepare_data.py
```

This regenerates `data/annotated/` with an 80/20 train/val split and prints
a per-class instance summary. If any class has fewer than 30 instances the
script prints a warning.

---

## 3. Training on Google Colab (Recommended)

1. Open `notebooks/colab_train.ipynb` in Google Colab.
2. Upload `data/annotated/` as a ZIP **or** mount Google Drive with the data.
3. Run all cells — training uses a free T4 GPU with these defaults:
   - `model='yolov8n.pt'`, `epochs=100`, `imgsz=640`, `batch=16`
   - Augmentations: mosaic, mixup, HSV jitter, rotation.
4. Download the resulting `best.pt` and place it at `models/detector_best.pt`.

### CPU fallback (slow)

```bash
python src/train_detector.py
```

This trains for 30 epochs on CPU. Expect it to be **very** slow on a laptop.

---

## 4. Running Inference

### Single image

```bash
python src/main.py --source path/to/image.jpg
```

### Webcam (live)

```bash
python src/main.py --source webcam
```

### Video file

```bash
python src/main.py --source path/to/video.mp4
```

Press **q** to stop webcam / video processing.

Annotated output is saved to `output/annotated_video.mp4` (video) or
`output/annotated_image.jpg` (image). Violations are logged to
`output/violations.csv`.

---

## 5. Running the Dashboard

```bash
streamlit run src/dashboard.py
```

Or use the convenience script:

```bash
# Windows
run_dashboard.bat

# Linux / macOS
bash run_dashboard.sh
```

The dashboard provides:
- **Sidebar:** upload an image, capture from webcam, or upload a video.
- **Main area:** annotated result from the pipeline.
- **Violations table:** recent entries from `output/violations.csv`.
- **Metrics row:** total vehicles today, violations today, compliance rate.
- **Download button:** export today's violations as CSV.

---

## 6. Known Limitations

| Area              | Limitation                                                            |
|-------------------|-----------------------------------------------------------------------|
| CPU speed         | Real-time video inference is not feasible on most laptops.            |
| Small dataset     | Default training data (~400 images) is too small for production use.  |
| OCR               | PaddleOCR may struggle with angled / blurred plates.                  |
| Attribute models  | Type and colour classifiers are trained on synthetic data by default. |
| Tracking          | IoU tracker has no motion model; fast-moving vehicles may lose IDs.   |
| Night / rain      | No domain-adaptation; accuracy drops in adverse conditions.           |
