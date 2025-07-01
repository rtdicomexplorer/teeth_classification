import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob

# Pfade
model_path = "runs/train/teeth-classification/weights/best.pt"
val_images_dir = "dataset/images/val"
val_labels_dir = "dataset/labels/val"

# Modell laden
model = YOLO(model_path)

# Hilfsfunktion, um YOLO-Labels zu lesen
def read_yolo_labels(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

# Class Names aus data.yaml (hier anpassen falls nötig)
class_names = ["Incisors", "Canines", "Premolars", "Molars"]

# Liste der Validierungsbilder
image_paths = glob.glob(os.path.join(val_images_dir, "*.png"))

for img_path in image_paths[:5]:  # Hier nur 5 Bilder zum schnellen Test
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Ground-Truth laden
    label_path = os.path.join(val_labels_dir, os.path.basename(img_path).replace(".png", ".txt"))
    gt_boxes = read_yolo_labels(label_path)

    # Vorhersage vom Modell
    results = model.predict(img_path, imgsz=800)
    preds = results[0]

    # Zeichne Ground-Truth (blau)
    gt_img = img_rgb.copy()
    for class_id, x_c, y_c, bw, bh in gt_boxes:
        x1 = int((x_c - bw/2)*w)
        y1 = int((y_c - bh/2)*h)
        x2 = int((x_c + bw/2)*w)
        y2 = int((y_c + bh/2)*h)
        cv2.rectangle(gt_img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(gt_img, class_names[class_id], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Zeichne Prediction (grün)
    pred_img = img_rgb.copy()
    for *box, score, class_id in preds.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, box)
        cls = int(class_id)
        cv2.rectangle(pred_img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(pred_img, f"{class_names[cls]} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Zeige nebeneinander
    combined = cv2.hconcat([gt_img, pred_img])
    plt.figure(figsize=(15,8))
    plt.title(f"Ground-Truth (links) vs. Prediction (rechts): {os.path.basename(img_path)}")
    plt.imshow(combined)
    plt.axis('off')
    plt.show()
