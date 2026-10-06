"""Data Preparation and Formatting for SignalGuard.

This module processes raw vehicle and license plate datasets into YOLO-format
annotations, splits images into train/val sets, and validates labels.
"""

import os
import sys
import random
import shutil
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# Mapping of class names to IDs
CLASS_MAP = {"vehicle": 0, "number_plate": 1, "person": 2}

def _load_annotations_xml(xml_path: Path):
    """Parse a Pascal VOC XML file and return list of (class_id, bbox) tuples.
    bbox is (xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name not in CLASS_MAP:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objs.append((CLASS_MAP[name], (xmin, ymin, xmax, ymax)))
    return objs

def _load_annotations_coco(coco_path: Path, image_id: int):
    """Extract annotations for a given image ID from a COCO JSON file.
    Returns list of (class_id, bbox) where bbox is (xmin, ymin, xmax, ymax).
    """
    data = json.load(open(coco_path))
    cat_id_to_name = {cat['id']: cat['name'].lower() for cat in data.get('categories', [])}
    anns = []
    for ann in data.get('annotations', []):
        if ann['image_id'] != image_id:
            continue
        cat_name = cat_id_to_name.get(ann['category_id'])
        if cat_name not in CLASS_MAP:
            continue
        x, y, w, h = ann['bbox']
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)
        anns.append((CLASS_MAP[cat_name], (xmin, ymin, xmax, ymax)))
    return anns

def _get_image_size(img_path: Path):
    """Obtain image width and height using OpenCV if available, otherwise default to 1x1.
    """
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Failed to read image")
        return img.shape[1], img.shape[0]
    except Exception:
        return 1, 1

def _convert_to_yolo(img_path: Path, objects, label_path: Path):
    """Write YOLO formatted label file for given objects.
    objects: list of (class_id, (xmin, ymin, xmax, ymax)).
    """
    img_w, img_h = _get_image_size(img_path)
    lines = []
    for cls_id, (xmin, ymin, xmax, ymax) in objects:
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    label_path.write_text('\n'.join(lines))

def prepare_data():
    raw_dir = Path('data/raw')
    if not raw_dir.exists():
        print('Raw data directory not found. Run download_data first.')
        sys.exit(1)

    # Gather all image files
    image_paths = list(raw_dir.rglob('*.jpg')) + list(raw_dir.rglob('*.jpeg')) + list(raw_dir.rglob('*.png'))
    if not image_paths:
        print('No image files found in data/raw.')
        sys.exit(1)

    random.shuffle(image_paths)
    split_idx = int(0.8 * len(image_paths))
    train_imgs = image_paths[:split_idx]
    val_imgs = image_paths[split_idx:]

    # Destination directories
    train_img_dir = Path('data/annotated/train/images')
    train_lbl_dir = Path('data/annotated/train/labels')
    val_img_dir = Path('data/annotated/val/images')
    val_lbl_dir = Path('data/annotated/val/labels')
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def _process(img_list, img_dest, lbl_dest):
        for img_path in img_list:
            shutil.copy2(img_path, img_dest / img_path.name)
            base = img_path.stem
            xml_path = img_path.with_suffix('.xml')
            coco_path = img_path.with_suffix('.json')
            objects = []
            if xml_path.exists():
                objects = _load_annotations_xml(xml_path)
            elif coco_path.exists():
                try:
                    img_id = int(base)
                    objects = _load_annotations_coco(coco_path, img_id)
                except Exception:
                    objects = []
            lbl_path = lbl_dest / (base + '.txt')
            _convert_to_yolo(img_path, objects, lbl_path)

    _process(train_imgs, train_img_dir, train_lbl_dir)
    _process(val_imgs, val_img_dir, val_lbl_dir)

    # Create dataset.yaml
    yaml_content = (
        f"train: {train_img_dir.as_posix()}\n"
        f"val: {val_img_dir.as_posix()}\n"
        f"\n"
        f"nc: {len(CLASS_MAP)}\n"
        f"names: [{', '.join([f'\"{k}\"' for k in CLASS_MAP.keys()])}]\n"
    )
    Path('data/dataset.yaml').write_text(yaml_content)

    # Summary counts
    def _count_instances(label_dir: Path):
        counts = {cid: 0 for cid in CLASS_MAP.values()}
        for lbl_file in label_dir.rglob('*.txt'):
            for line in lbl_file.read_text().splitlines():
                if not line.strip():
                    continue
                cid = int(line.split()[0])
                counts[cid] += 1
        return counts

    train_counts = _count_instances(train_lbl_dir)
    val_counts = _count_instances(val_lbl_dir)
    total_counts = {cid: train_counts[cid] + val_counts[cid] for cid in CLASS_MAP.values()}

    print('--- Summary ---')
    print(f'Total images: {len(image_paths)} (train: {len(train_imgs)}, val: {len(val_imgs)})')
    print('Instances per class (train):')
    for name, cid in CLASS_MAP.items():
        print(f'  {name}: {train_counts[cid]}')
    print('Instances per class (val):')
    for name, cid in CLASS_MAP.items():
        print(f'  {name}: {val_counts[cid]}')
    print('Instances per class (total):')
    for name, cid in CLASS_MAP.items():
        print(f'  {name}: {total_counts[cid]}')

    for name, cid in CLASS_MAP.items():
        if total_counts[cid] < 30:
            print(f'WARNING: Class "{name}" has only {total_counts[cid]} instances. Consider adding more images.')

if __name__ == "__main__":
    prepare_data()

