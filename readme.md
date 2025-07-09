# 🦷 Teeth classification using YOLO and UNET


### Enviroment preparation:
- python -m venv venv 
- venv\Scripts\activate
- pip install -r requirements.txt


### Dataset

-  "kvipularya/a-collection-of-dental-x-ray-images-for-analysis" on https://www.kaggle.com/datasets/kvipularya/a-collection-of-dental-x-ray-images-for-analysis
- - a method to download the dataset is already provided : (labelme_utilities/get_kaggle_dataset) of course you need a kaggle token (kaggle.json)
- - we will use just the images (all-imaages) and the labelme-labels, we will create from those the yolo-labels and unet-masks...
- Data Structure:
- - all-images: p1.png->p73.png   panoramic data
- - mask labelme-labels: p1.json -> p73.json mask in labelme format (json)


## STEPs:
### let run data_preparation.py to prepare the dataset
- 1 - recognize the classification from labelme-labels (labelme_utilities/get_dominant_labels)
- - we will find 4 classes [1: Incisors; 2:Canines ; 3: Premolars; 4: Molars]
- 2 - create the unet-masks from labelme (contain the 5 possible values 0-4,multiclassification)
- 3 - create the yolo-labels from labelme
- 4 - split the whole dataset (images, labels, masks) into train, val
- 5 - create the yaml file to execute the yolo training
- 6 - let run train_yolo.py for the yolo training (input: output_fir, epochs )
- 7 - let run train_unet.py for the unet training (input: output_fir, epochs )
- 8 - visualize the results (visualize_results)


The mmodel will be saved into weights







interesting for the dataset also https://github.com/devichand579/Instance_seg_teeth

here the new dataset: https://universe.roboflow.com/rf100-vl-fsod/ufba-425-asgxh-fsod-djrs

the classes (channels) ['11','12','13','14','15','16','17','18',
                        '21','22','23','24','25','26','27','28',
                        '31','32','33','34','35','36','37','38',
                        '41','42','43','44','45','46','47','48']

the labes are splitted in folder cate1...cate10 to distinguish the charateristic..
in each folder they are saved as: cate(i)-img_channel_nr. Each label contain just one channel, then the relation is: 
### 1-image-> n-channels-labels. single class

To train a model, we can use the dataset with these structure or we can merge for each cate all channels-label of one image i just one label also 
### 1-image->onelabel(with all channels) multiclass



create new label:

https://github.com/wkentaro/labelme


https://www.cvat.ai/