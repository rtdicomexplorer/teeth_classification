import os
from pathlib import Path
from labelme_utilities import get_dominant_labels, create_data_yaml, convert_labelme_to_yolo,split_dataset, get_kaggle_dataset, create_all_masks, split_dataset
import sys

# step 0: just if you have a token of kaggle
# os.makedirs(os.path.join(final_dataset,unet_labels), exist_ok=True)
# get_kaggle_dataset()


def main(root):
    labelme_dir = os.path.join(root,"labelme-labels")
    image_dir = os.path.join(root,"all-images")
    yolo_labels = "labelme-yolo-labels"
    unet_labels = "labelme-unet-labels"
    final_dataset = "dataset"
    os.makedirs(final_dataset, exist_ok=True)

    # step 1: get the classification
    group_to_label, label_to_id, label_list = get_dominant_labels(labelme_dir)

    # step 2: create the unet-masks from labelme 
    os.makedirs(unet_labels, exist_ok=True)
    create_all_masks(Path(image_dir), Path(labelme_dir),Path(unet_labels),create_preview=False)

    # step 3: create the yolo-labels from labelme
    convert_labelme_to_yolo(labelme_dir, image_dir, yolo_labels, group_to_label, label_to_id)

    # step 4: split the whole dataset (images, labels, masks) into train, val
    split_dataset(image_dir, yolo_labels,unet_labels, final_dataset)

    # step 6: create the yaml file to execute the yolo training
    yaml_path = os.path.join(final_dataset, "data.yaml")
    create_data_yaml(yaml_path, os.path.join(final_dataset, "images/train"), os.path.join(final_dataset, "images/val"), label_list)

    # step 7: start the training..


if __name__ == "__main__":
    root = sys.argv[1:][0] 
    #  root = "kaggle/dataset"
    main(root)
