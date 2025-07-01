import os
from labelme_utilities import get_dominant_labels, create_data_yaml, convert_labelme_to_yolo,split_dataset

# Ordnerpfade

json_dir = "labelme-labels"
image_dir = "all-images"
output_labels = "labelme-yolo-labels"
final_dataset = "dataset"
os.makedirs(final_dataset, exist_ok=True)

yaml_path = os.path.join(final_dataset, "data.yaml")

# Schritte ausführen
group_to_label, label_to_id, label_list = get_dominant_labels(json_dir)
create_data_yaml(yaml_path, os.path.join(final_dataset, "images/train"), os.path.join(final_dataset, "images/val"), label_list)
convert_labelme_to_yolo(json_dir, image_dir, output_labels, group_to_label, label_to_id)
split_dataset(image_dir, output_labels, final_dataset)