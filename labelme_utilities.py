import json
import os
import numpy as np
from collections import defaultdict
import cv2
from PIL import Image
from pathlib import Path
import shutil
import random
import kagglehub


def get_dominant_labels(json_dir):
    group_label_freq = defaultdict(lambda: defaultdict(int))  # group_id → label → count

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                for shape in data.get("shapes", []):
                    gid = shape.get("group_id")
                    label = shape.get("label")
                    if gid is not None and label:
                        group_label_freq[gid][label] += 1

    group_to_label = {}
    label_set = set()
    for gid, labels in group_label_freq.items():
        dominant = max(labels.items(), key=lambda x: x[1])[0]
        group_to_label[gid] = dominant
        label_set.add(dominant)

    label_list = sorted(list(label_set))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    return group_to_label, label_to_id, label_list

def create_data_yaml(save_path, train_dir, val_dir, label_list):
    yaml_content = {
        'train': os.path.abspath(train_dir),
        'val': os.path.abspath(val_dir),
        'nc': len(label_list),
        'names': label_list
    }

    with open(save_path, 'w') as f:
        import yaml
        yaml.dump(yaml_content, f)

    print(f"✅ data.yaml saved: {save_path}")

def convert_labelme_to_yolo(json_dir, image_dir, output_label_dir, group_to_label, _label_to_id):

    # some labelme have typo error with plurals (s)
    label_to_id = {k.lower(): v for k, v in _label_to_id.items()}
    os.makedirs(output_label_dir, exist_ok=True)
    skipped_files = 0
    converted_files = 0
    unknown_labels = set()

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        json_path = os.path.join(json_dir, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = os.path.join(image_dir, data["imagePath"])
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines = []

        for shape in data.get("shapes", []):
            group_id = shape.get("group_id")
            label = group_to_label.get(group_id) if group_id is not None else shape.get("label")

            # Normalize label (e.g., case-insensitive, remove trailing 's')
            label = label.strip().lower()
            label = label.rstrip('s') + 's'  # Ensure plural form: 'molar' → 'molars'

            if label not in label_to_id:
                unknown_labels.add(label)
                continue

            class_id = label_to_id[label]
            points = shape["points"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # Normalize box coordinates
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bbox_width = (xmax - xmin) / w
            bbox_height = (ymax - ymin) / h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")



        output_file = os.path.join(output_label_dir, filename.replace(".json", ".txt"))
        if yolo_lines:
            with open(output_file, "w") as out:
                out.write("\n".join(yolo_lines))
            converted_files += 1
        else:
            # Don't write empty label files
            skipped_files += 1
    print(f"✅ Conversion complete. {converted_files} files converted, {skipped_files} skipped (no valid labels).")

    

def __split_dataset(image_dir, label_dir, out_dir, train_ratio=0.8):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for split, file_list in [("train", train_files), ("val", val_files)]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
        for file in file_list:
            shutil.copy(os.path.join(image_dir, file), os.path.join(out_dir, "images", split, file))
            label_file = file.replace(".png", ".txt")
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(out_dir, "labels", split, label_file))

    print("✅ Train/Val Split completed")


def split_dataset(image_dir, yolo_label_dir, unet_label_dir, out_dir, train_ratio=0.8):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for split, file_list in [("train", train_files), ("val", val_files)]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)#yolo need a correspondence between image and labels name should bve labels...
        os.makedirs(os.path.join(out_dir, "masks", split), exist_ok=True)
        for file in file_list:
            shutil.copy(os.path.join(image_dir, file), os.path.join(out_dir, "images", split, file))
            yolo_label_file = file.replace(".png", ".txt")
            shutil.copy(os.path.join(yolo_label_dir, yolo_label_file), os.path.join(out_dir, "labels", split, yolo_label_file))
            shutil.copy(os.path.join(unet_label_dir, file), os.path.join(out_dir, "masks", split, file))

    print("✅ Train/Val Split completed")


def get_kaggle_dataset(dataset_path= "kvipularya/a-collection-of-dental-x-ray-images-for-analysis", kaggle_path= 'kaggle'):
    '''
    Method to download the trainings data. you have got already a kaggle token.
    '''
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_path
    # Download the latest version of a dataset
    path = kagglehub.dataset_download(dataset_path)
    print("Data downloaded to:", path)
    destination_path = os.path.join(kaggle_path, 'dataset')
    os.makedirs(destination_path, exist_ok=True)
    shutil.copytree(path, destination_path, dirs_exist_ok=True)
    print("Copied data to:", destination_path)
    shutil.rmtree(path)


def __labelme_json_to_mask(json_path, img_shape):
    class_map = {
        "Molars": 1,
        "Premolars": 2,
        "Incisors": 3
    }

    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros(img_shape, dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'])
        points = np.round(points).astype(np.int32)
        class_idx = class_map.get(label, 0)
        cv2.fillPoly(mask, [points], class_idx)

    return mask


def create_all_masks(image_dir, json_dir, unet_labels_dir, create_preview = True):

    for img_path in sorted(image_dir.glob("*.png")):
        json_path = json_dir / (img_path.stem + ".json")

        if not json_path.exists():
            print(f"⚠️ JSON missed for {img_path.name}.")
            continue

        img = Image.open(img_path)
        img_shape = (img.height, img.width)

        mask = __labelme_json_to_mask(json_path, img_shape)
        
        mask_path = Path(unet_labels_dir) / (img_path.stem + ".png")
        Image.fromarray(mask).save(mask_path)

        if create_preview == True:
            color_mask = __convert_mask_to_color(mask)
            preview_dir = Path("preview_masks")
            preview_dir.mkdir(exist_ok=True)
            Image.fromarray(color_mask).save(preview_dir / f"{img_path.stem}_color.png")
            unique_values = np.unique(mask)
            print(f"{img_path.name} → classes in mask: {unique_values}")


    print("✅ All masks saved")


def __convert_mask_to_color(mask):
    """
    Wandelt eine Label-Maske (0=Background, 1=Molars, ...) in ein RGB-Bild um.
    """
    color_map = {
        0: [0, 0, 0],         # Hintergrund = schwarz
        1: [255, 0, 0],       # Molars = rot
        2: [0, 255, 0],       # Premolars = grün
        3: [0, 0, 255]        # Incisors = blau
    }

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in color_map.items():
        color_mask[mask == class_idx] = color

    return color_mask

