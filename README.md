# SignalGuard 🚦🛡️

SignalGuard is an automated computer vision traffic monitoring and compliance system. It detects vehicles at traffic signals from images or video feeds, recognizes and reads license plates, validates plate compliance against official government formatting standards, tracks non-compliant vehicles across frames, and extracts vehicle attributes (type: 2/3/4-wheeler, color, and estimated occupants). Violations are logged to CSV and preserved with photographic evidence.

## Hardware & Architecture Highlights
- Local CPU-friendly inference optimized with frame-skipping and lightweight models.
- Cloud/Colab GPU acceleration workflow for model training.
- Modular architecture adhering to single-responsibility design.

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Verify your environment setup:

```bash
python src/check_env.py
```

## Directory Structure
```
SignalGuard/
├── data/
│   ├── raw/
│   ├── annotated/
│   └── dataset.yaml
├── models/
├── src/
│   ├── __init__.py
│   ├── check_env.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train_detector.py
│   ├── train_attribute_models.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── compliance.py
│   ├── tracker.py
│   ├── output_handler.py
│   └── main.py
├── notebooks/
│   └── colab_train.ipynb
├── output/
│   └── violations/
├── requirements.txt
├── README.md
└── .gitignore
```
