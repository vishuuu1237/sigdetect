# SignalGuard 🚦🛡️

**Automated traffic monitoring & license-plate compliance system powered by
YOLOv8, PaddleOCR, and MobileNetV2.**

SignalGuard detects vehicles at traffic signals from images or video feeds,
recognises and reads license plates, validates plate compliance against official
Indian formatting standards, tracks non-compliant vehicles across frames, and
extracts vehicle attributes (type, colour, occupant count). Violations are
logged to CSV and preserved with photographic evidence.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│         Image / Video / Webcam  ──►  frame_skip sampler          │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              YOLOv8 DETECTOR  (vehicle + plate)                  │
│              models/detector_best.pt                             │
└────┬──────────────────────┬──────────────────────────────────────┘
     │                      │
     ▼                      ▼
┌────────────┐     ┌────────────────┐
│ Vehicle    │     │ Plate crop     │
│ crop       │     │  ► PaddleOCR   │
└────┬───────┘     │  ► Compliance  │
     │             └───────┬────────┘
     ▼                     │
┌────────────────┐         │
│ MobileNetV2    │         │
│ Type + Colour  │         │
│ classifiers    │         │
└────┬───────────┘         │
     │                     │
     ▼                     │
┌────────────────┐         │
│ YOLOv8 COCO   │         │
│ Person detect  │         │
│ (occupants)    │         │
└────┬───────────┘         │
     │                     │
     ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    IoU TRACKER                                    │
│        Maintains track_id → {plate, type, colour, occupants}     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   OUTPUT HANDLER                                  │
│   output/violations.csv  ·  output/violations/<id>/vehicle.jpg   │
│   Annotated frames  ·  Streamlit dashboard                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/your-username/sigdetect.git
cd sigdetect
pip install -r requirements.txt

# 2. Verify environment
python src/check_env.py

# 3. Download & prepare data
python src/download_data.py
python src/prepare_data.py

# 4. Train detector (CPU fallback — use Colab for full training)
python src/train_detector.py

# 5. Run inference on an image
python src/main.py --source path/to/image.jpg

# 6. Run inference on webcam
python src/main.py --source webcam

# 7. Launch the dashboard
streamlit run src/dashboard.py
```

---

## Directory Structure

```
sigdetect/
├── data/
│   ├── raw/                  # Downloaded images + annotations
│   ├── annotated/            # YOLO-format train/val split
│   │   ├── train/images/
│   │   ├── train/labels/
│   │   ├── val/images/
│   │   └── val/labels/
│   └── dataset.yaml
├── docs/
│   ├── USAGE.md              # Full usage instructions
│   └── DATA_GUIDE.md         # Data collection & annotation guide
├── models/
│   ├── detector_best.pt      # Trained YOLOv8 detector
│   ├── type_classifier.pt    # Vehicle-type MobileNetV2
│   └── color_classifier.pt   # Vehicle-colour MobileNetV2
├── notebooks/
│   └── colab_train.ipynb     # One-click Colab training notebook
├── output/
│   ├── violations/           # Per-violation folders with crops
│   └── violations.csv        # Master violation log
├── src/
│   ├── check_env.py          # Environment verification
│   ├── download_data.py      # Dataset downloader (kagglehub)
│   ├── prepare_data.py       # Annotation converter + splitter
│   ├── train_detector.py     # CPU training fallback
│   ├── train_attribute_models.py  # Type + colour classifiers
│   ├── evaluate.py           # mAP / precision / recall
│   ├── pipeline.py           # Full inference orchestration
│   ├── compliance.py         # Indian plate regex + heuristics
│   ├── tracker.py            # IoU-based multi-object tracker
│   ├── output_handler.py     # Violation logger + image saver
│   ├── dashboard.py          # Streamlit interactive dashboard
│   └── main.py               # CLI entry point
├── tests/
│   ├── test_compliance.py
│   ├── test_tracker.py
│   └── test_pipeline.py
├── run_dashboard.bat          # Windows launcher
├── run_dashboard.sh           # Linux/macOS launcher
├── requirements.txt
└── README.md
```

---

## Hardware & Architecture Highlights

- **CPU-friendly inference** optimised with frame-skipping and lightweight models.
- **Colab GPU training** workflow via `notebooks/colab_train.ipynb`.
- **Modular architecture** — each module has a single responsibility and can be
  swapped independently.

---

## Documentation

| Document                                          | Description                          |
|---------------------------------------------------|--------------------------------------|
| [docs/USAGE.md](docs/USAGE.md)                   | Installation, training, inference    |
| [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md)         | Data collection & annotation guide   |
| [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) | Google Colab training notebook |

---

## License

MIT
# sigdetect
