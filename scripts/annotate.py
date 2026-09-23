"""
Two-pass annotation pipeline for PCINet training images.

Pass 1 — Relevant (Grounding DINO tiny):
    Prompt: "building and wall and gate."
    Central-region fallback (5% inset) when nothing is detected.

Pass 2 — Irrelevant (YOLOv8m):
    COCO classes 0/1/2/3/5/7 (person, bicycle, car, motorcycle, bus, truck) @ conf=0.25

Outputs (in OUTPUT_DIR):
    labels/              PASCAL VOC XML, one per image
    annotations_train.csv  flat CSV: image_name, box_type, xmin, ymin, xmax, ymax

Resume-safe: images with an existing XML are skipped on re-run.

Usage:
    python scripts/annotate_train.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = r"..."
IMAGES_DIR  = os.path.join(DATASET_DIR, "images")
TRAIN_CSV   = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV     = os.path.join(DATASET_DIR, "val.csv")

# Where to write labels/ and annotations_train.csv
OUTPUT_DIR  = r"..."

# Set True to also annotate val.csv images (804 extra)
INCLUDE_VAL = False

# Grounding DINO — exact settings used for test-set annotation
GDINO_MODEL     = "IDEA-Research/grounding-dino-tiny"
GDINO_PROMPT    = "building and wall and gate."
GDINO_THRESHOLD = 0.30

# YOLOv8m — downloads automatically to ultralytics cache if not present
YOLO_MODEL_PATH = "yolov8m.pt"
YOLO_CONF       = 0.25
IRRELEVANT_IDS  = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
# ============================================================

LABELS_DIR     = os.path.join(OUTPUT_DIR, "labels")
ANNOT_CSV_PATH = os.path.join(OUTPUT_DIR, "annotations_train.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_voc_xml(xml_path, img_name, img_path, width, height, boxes):
    """boxes: list of (box_type, xmin, ymin, xmax, ymax)"""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text   = "images"
    ET.SubElement(root, "filename").text = img_name
    ET.SubElement(root, "path").text     = img_path
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text  = str(width)
    ET.SubElement(sz, "height").text = str(height)
    ET.SubElement(sz, "depth").text  = "3"
    for btype, x1, y1, x2, y2 in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text      = btype
        ET.SubElement(obj, "pose").text      = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    lines = pretty.split("\n")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write("\n".join(lines[1:]))


def gdino_relevant_boxes(pil_img, width, height, processor, model):
    inputs = processor(
        images=pil_img, text=GDINO_PROMPT, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=GDINO_THRESHOLD,
        text_threshold=GDINO_THRESHOLD,
        target_sizes=[(height, width)],
    )[0]
    boxes = [
        (max(0, int(x1)), max(0, int(y1)),
         min(width - 1, int(x2)), min(height - 1, int(y2)))
        for x1, y1, x2, y2 in results["boxes"].cpu().numpy()
    ]
    if not boxes:
        # Central-region fallback: 5% inset on every side (matches test-set annotation)
        boxes = [(
            int(width * 0.05), int(height * 0.05),
            int(width * 0.95), int(height * 0.95),
        )]
    return boxes


def yolo_irrelevant_boxes(img_path, width, height, model):
    results = model(img_path, conf=YOLO_CONF, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) in IRRELEVANT_IDS:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append((
                max(0, int(x1)), max(0, int(y1)),
                min(width - 1, int(x2)), min(height - 1, int(y2)),
            ))
    return boxes


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    # ── Collect image names ───────────────────────────────────────────────────
    image_names = pd.read_csv(TRAIN_CSV)["name"].tolist()
    if INCLUDE_VAL:
        image_names += pd.read_csv(VAL_CSV)["name"].tolist()
    image_names = list(dict.fromkeys(image_names))  # deduplicate, preserve order

    n_total = len(image_names)
    print(f"Device  : {device}")
    print(f"Images  : {n_total}  ({'train+val' if INCLUDE_VAL else 'train only'})")
    print(f"Output  : {OUTPUT_DIR}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading Grounding DINO...")
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
    gdino_model.eval()
    print("Grounding DINO ready.")

    print("Loading YOLOv8m...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8m ready.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    n_done = n_skipped = n_fallback = 0
    csv_rows = []

    for i, img_name in enumerate(image_names):
        stem     = os.path.splitext(img_name)[0]
        xml_path = os.path.join(LABELS_DIR, stem + ".xml")
        img_path = os.path.join(IMAGES_DIR, img_name)

        # Resume: XML already exists — read it for the CSV, skip detection
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            for obj in tree.getroot().findall("object"):
                bb = obj.find("bndbox")
                csv_rows.append({
                    "image_name": img_name,
                    "box_type":   obj.find("name").text,
                    "xmin": int(bb.find("xmin").text),
                    "ymin": int(bb.find("ymin").text),
                    "xmax": int(bb.find("xmax").text),
                    "ymax": int(bb.find("ymax").text),
                })
            n_skipped += 1
            continue

        if not os.path.exists(img_path):
            print(f"WARNING: not found — {img_name}")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        width, height = pil_img.size

        # Pass 1: relevant
        rel_boxes = gdino_relevant_boxes(pil_img, width, height, gdino_processor, gdino_model)
        fallback_box = (int(width*0.05), int(height*0.05), int(width*0.95), int(height*0.95))
        if rel_boxes == [fallback_box]:
            n_fallback += 1

        # Pass 2: irrelevant
        irr_boxes = yolo_irrelevant_boxes(img_path, width, height, yolo_model)

        all_boxes = []
        for x1, y1, x2, y2 in rel_boxes:
            all_boxes.append(("relevant", x1, y1, x2, y2))
            csv_rows.append({"image_name": img_name, "box_type": "relevant",
                             "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})
        for x1, y1, x2, y2 in irr_boxes:
            all_boxes.append(("irrelevant", x1, y1, x2, y2))
            csv_rows.append({"image_name": img_name, "box_type": "irrelevant",
                             "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})

        write_voc_xml(xml_path, img_name, img_path, width, height, all_boxes)
        n_done += 1

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            pct = 100 * (i + 1) / n_total
            print(f"[{pct:5.1f}%]  {i+1}/{n_total}  "
                  f"done={n_done}  skipped={n_skipped}  fallbacks={n_fallback}",
                  flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    annot_df = pd.DataFrame(csv_rows, columns=["image_name", "box_type",
                                                "xmin", "ymin", "xmax", "ymax"])
    annot_df.to_csv(ANNOT_CSV_PATH, index=False)

    print(f"\nDone.")
    print(f"  Annotated  : {n_done}")
    print(f"  Skipped    : {n_skipped}  (already had XML)")
    print(f"  Fallbacks  : {n_fallback}  (no structure detected by GroundingDINO)")
    print(f"  Total boxes: {len(annot_df)}")
    print(f"  CSV saved  : {ANNOT_CSV_PATH}")


if __name__ == "__main__":
    main()
