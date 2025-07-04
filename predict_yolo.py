import cv2
import os
from ultralytics import YOLO
from matplotlib import pyplot as plt
import sys


import json

import csv

def __save_predictions_as_csv(results, output_file, class_names):
    rows = [["image_name", "class_id", "class_name", "x1", "y1", "x2", "y2", "confidence"]]

    for result in results:
        image_name = os.path.basename(result.path)
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append([image_name, cls_id, class_names[cls_id], x1, y1, x2, y2, conf])

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ Saved CSV predictions to {output_file}")



def __save_predictions_as_json(results, output_file, class_names):
    data = {}

    for result in results:
        image_name = os.path.basename(result.path)
        predictions = []
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # absolute pixel coords
            predictions.append({
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        data[image_name] = predictions

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ Saved JSON predictions to {output_file}")

def __save_predictions_as_txt(results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    for result in results:
        image_name = os.path.basename(result.path)
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)

        lines = []
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x_center, y_center, w, h = box.xywhn[0].tolist()  # normalized coords
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f}")

        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines))

    print(f"✅ Saved YOLO txt predictions in {output_dir}")


def __save_predictionresult_as_images(model,results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    class_colors = {
        'Canines': (0, 255, 0),      # Green
        'Incisors': (255, 0, 0),     # Blue
        'Molars': (0, 0, 255),       # Red
        'Premolars': (255, 255, 0)   # Yellow
}

    for result in results:
        im_array = result.orig_img.copy()

        # Draw boxes with confidence only
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = class_colors.get(label, (128, 128, 128))

            overlay = im_array.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, im_array, 1 - alpha, 0, im_array)

            cv2.putText(im_array, f'{conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Draw vertical legend on the image ---

        start_x, start_y = 10, 10
        box_size = 20
        spacing_y = 8
        font_scale = 0.6
        thickness = 1

        # Calculate legend box size
        max_text_width = 0
        for cls_name in class_colors.keys():
            text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            if text_size[0] > max_text_width:
                max_text_width = text_size[0]

        legend_width = box_size + 10 + max_text_width + 10
        legend_height = (box_size + spacing_y) * len(class_colors) + 5

        # Draw background rectangle (white, semi-transparent)
        overlay = im_array.copy()
        cv2.rectangle(overlay, (start_x-5, start_y-5), (start_x + legend_width, start_y + legend_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, im_array, 0.3, 0, im_array)

        # Draw each class with colored box and label vertically
        y = start_y
        for cls_name, color in class_colors.items():
            # Colored box
            cv2.rectangle(im_array, (start_x, y), (start_x + box_size, y + box_size), color, -1)
            # Text label (right to the box)
            cv2.putText(im_array, cls_name, (start_x + box_size + 8, y + box_size - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
            y += box_size + spacing_y

        # Save image
        filename = os.path.basename(result.path)
        out_path = os.path.join(output_dir, f"overlay_{filename}")
        cv2.imwrite(out_path, im_array)

    print(f"✅ Saved {len(results)} images with vertical legend overlays to: {output_dir}")


def predict(model_name,source_dir):
    # Load model and predict
    model = YOLO(model_name)
    results = model.predict(source=source_dir, conf=0.5)

    __save_predictionresult_as_images(model,results,os.path.join(source_dir,'yolo_predictions_mask_overlay'))
    class_names = ['Canines', 'Incisors', 'Molars', 'Premolars']

    __save_predictions_as_txt(results, os.path.join(source_dir,'predictions_txt'))
    __save_predictions_as_json(results, os.path.join(source_dir,'predictions.json'), class_names)
    __save_predictions_as_csv(results, os.path.join(source_dir,'predictions.csv'), class_names)


if __name__=="__main__":
    ''' Input parameter: model name and source dir to analize'''
    predict(sys.argv[1:][0],sys.argv[1:][1])