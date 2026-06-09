#!/usr/bin/env python3
"""Prepare image folder structure for ImageFolder-style inference.
Usage:
  python scripts/prepare_images.py --images /path/to/images --labels /path/to/labels.csv --out streetview/my_images --copy

labels.csv should have two columns: name,label with a header or without.
'name' can be basename (file.jpg) or relative/absolute path. 'label' should be integer or string class.
By default the script will move files; use --copy to copy instead.
"""
import argparse
import os
import shutil
import csv
from pathlib import Path


def load_label_map(csv_path):
    label_map = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return label_map
        # detect header
        if len(header) >= 2 and (header[0].lower() == 'name' or header[1].lower() == 'label'):
            pass
        else:
            # first row was data
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            label = row[1].strip()
            label_map[name] = label
    return label_map


def find_image(images_root, name):
    p = Path(name)
    # absolute
    if p.is_absolute() and p.exists():
        return p
    # basename search
    for root, _, files in os.walk(images_root):
        if p.name in files:
            return Path(root) / p.name
    # try relative to images_root
    candidate = Path(images_root) / name
    if candidate.exists():
        return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Path to folder containing images (recursive search)')
    ap.add_argument('--labels', required=True, help='CSV file with columns name,label')
    ap.add_argument('--out', default='streetview/my_images', help='Output root for ImageFolder (class subfolders)')
    ap.add_argument('--copy', action='store_true', help='Copy files instead of moving')
    ap.add_argument('--missing-log', default='missing_images.txt', help='Write missing filenames to this file')
    args = ap.parse_args()

    images_root = args.images
    labels_csv = args.labels
    out_root = args.out
    do_copy = args.copy

    os.makedirs(out_root, exist_ok=True)

    label_map = load_label_map(labels_csv)
    missing = []
    moved = 0

    for name, label in label_map.items():
        src = find_image(images_root, name)
        if src is None:
            missing.append(name)
            continue
        class_dir = Path(out_root) / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / Path(name).name
        try:
            if do_copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            print(f'Error moving {src} -> {dst}: {e}')
    
    with open(args.missing_log, 'w', encoding='utf-8') as f:
        for m in missing:
            f.write(m + '\n')

    print(f'Done. Files placed: {moved}. Missing: {len(missing)}. Missing list: {args.missing_log}')

if __name__ == '__main__':
    main()
