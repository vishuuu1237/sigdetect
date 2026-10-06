# SignalGuard — Data Collection & Annotation Guide

## 1. Overview

SignalGuard requires labelled images with bounding boxes for three classes:

| Class ID | Label          | Description                            |
|----------|----------------|----------------------------------------|
| 0        | `vehicle`      | Car, bus, truck, auto-rickshaw, bike   |
| 1        | `number_plate` | Visible license / registration plate   |
| 2        | `person`       | Occupant visible inside or on vehicle  |

**Recommended minimum: 300+ images per class** for acceptable mAP. More is
always better — 1 000+ images per class is ideal for production deployment.

---

## 2. Annotation Tools

### Roboflow (web-based, recommended for beginners)

1. Create a free account at [roboflow.com](https://roboflow.com).
2. Create a new project → Object Detection → set the three classes above.
3. Upload images and draw bounding boxes.
4. Export in **YOLOv8** format (`.txt` label files).
5. Download and unzip into `data/annotated/`.

### LabelImg (offline, open source)

```bash
pip install labelImg
labelImg
```

1. Open Dir → select `data/raw/images/`.
2. Change save format to **YOLO** (button in the left toolbar).
3. Set save directory to `data/raw/labels/`.
4. Draw boxes, assign class names, save.
5. Run `python src/prepare_data.py` to split into train/val.

### CVAT (self-hosted, team use)

See [cvat.ai](https://www.cvat.ai) — export as YOLO 1.1.

---

## 3. Label Format (YOLO)

Each image `image_name.jpg` has a corresponding `image_name.txt`:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalised** to [0, 1] relative to image dimensions.

Example (`Cars0.txt`):

```
0 0.4812 0.5200 0.3100 0.4800
1 0.4900 0.6800 0.0800 0.0350
```

---

## 4. Adding Local / Regional Plates

Indian states use different plate prefixes (e.g. KA for Karnataka, MH for
Maharashtra). To improve OCR and compliance accuracy for your region:

1. **Collect 50+ images** of local plates from dashcam footage or publicly
   available traffic datasets.
2. **Annotate** the plate bounding boxes (class `1`).
3. **Update the regex** in `src/compliance.py` if your region uses a
   non-standard format. The default regex is:

   ```
   ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
   ```

4. Re-run training or fine-tune:

   ```bash
   python src/train_detector.py
   ```

### Plate variants to cover

| Type              | Example           | Notes                              |
|-------------------|-------------------|------------------------------------|
| Standard          | KA01AB1234        | Most common private vehicle plate  |
| Commercial (Y)    | KA01C1234         | Yellow background                  |
| Temporary         | KA01TC1234        | Red lettering                      |
| Electric vehicle  | KA01BH1234        | Green background (BH series)       |
| Diplomatic        | 84CD1234          | Blue plate, numeric state code     |

---

## 5. Data Augmentation Tips

The training pipeline already applies mosaic, mixup, HSV jitter, and rotation.
For additional robustness, consider:

- **Night-time images** — low light, headlamp glare.
- **Rain / fog** — reduced contrast, reflections.
- **Angled views** — plates seen from 30°–60° angles.
- **Motion blur** — fast-moving vehicles.

You can generate synthetic augmentations with Albumentations:

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 6. Quality Checklist

Before training, verify your dataset:

- [ ] Every image has a corresponding `.txt` label file.
- [ ] No label file is empty (run `python src/prepare_data.py` — it warns).
- [ ] Bounding boxes are tight (no excessive padding).
- [ ] Class distribution is roughly balanced (within 3× ratio).
- [ ] Images are at least 640×480; larger is fine (resized during training).
