# 🦷 Teeth classification

### preparation:
- python -m venv venv 
- venv\Scripts\activate
- pip install -r requirements.txt


### Dataset

- https://www.kaggle.com/datasets/kvipularya/a-collection-of-dental-x-ray-images-for-analysis

- Data Structure:
- -    all-images: p1.png->p73.png   panoramic data
- - mask labelme-labels: p1.json -> p73.json mask in labelme format




### class name
- 0 Canines
- 1 Incisors
- 2 Molars
- 3 Premolars

## TO DO:
- 1 - recognize the classification from labelme-labels (labelme_utilities/get_dominant_labels)
- 2 - create yolo data.yaml for the training using the above classification (labelme_utilities/create_data_yaml)
- 3 - Create yolo mask data from labelme (labelme_utilities/convert_labelme_to_yolo)
- 4 - split the images and the new yolo labels in train and valuation (80%- 20%) (labelme_utilities/split_dataset)
- 5 - train the model (train_yolo)
- 6 - visualize the results (visualize_results)




