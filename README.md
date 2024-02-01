### README ###

# ENVIRONMENT

- python: 3.7.9
- CUDA: V10.1.243

# USE CASE: damage similarity
The objective of this use case is:
- Given an input image and a gallery of images, find all the images of the gallery that contain similar damages to the one of the input (query) image.

# SOLUTION
The adopted solution is a feature-extraction network (osnet_x1_0) trained using a triplet loss:
- osnet_x1_0: Given an input image, a corresponding vector of features is generated;
- cosine similarity: the resulting features are compared so as to assess the similarity between them.

# DOWNLOAD
You can download the pre-trained model and dataset at [this link](https://drive.google.com/drive/u/1/folders/1VKMuGIejowfg7xylZsPKJ9VGdaxNWTYv).

# DATASET
Training:
- An overall number of 40331 images has been used during the training.

Testing:
- An overall number of 385 images has been used during the testing phase.


# OUTPUT AND METRICS

- mAP(mean Average Precision): 78.7%

- Rank-1  : 81.8%
- Rank-5  : 93.9%
- Rank-10 : 95.5%
- Rank-20 : 97.7%

INFERENCE TIME:
- 0.21 sec/image (cpu)
- 0.017 sec/image (gpu)

# INSTRUCTIONS

- create an environment with python 3.8 and install requirements.txt

- if you want to use tf similarity legacy notebooks, instead 
    - create an environment with python 3.8 
    - run conda install -c conda-forge cudnn=8.1 cudatoolkit=11.2
    - run pip install -r requirements_tfsim.txt
## TRAIN AND TEST

- open folder src
- set variables values inside config.py 
- open script utils.py
- set variables values inside setup_damage_detector() and setup_car_detector()
- open the folder notebook  
- use the notebook main.ipynb to use all the functionalities

## NOTEBOOKS
- generate_cropped_data_semisupervised.ipynb allows to label data using an available model 
- generate_cropped_data_supervised.ipynb allows to label data using .json annotations
- generate_dataset_encoding.ipynb allows to rename images in the format damageid_imageid.jpg (e.g. 4_5.jpg)
- generate_synthetic_dataset.ipynb allows to generate a synthetic dataset with spliced damages
- train.ipynb implements the training procedure 
- test_{framework}.ipynb implements the framework specific testing procedure 
- inference_similarity.ipynb implements the inference to give an insight of the model practical application
- visualize_activation_maps.ipynb allows to generate activation maps associated to the input images
- LEGACY_model_agnostic_test_tfsimil_PAPER_2.1 to replicate test with tf similarity library (worst than torch_reid as evinced from our tests) 
- model_agnostic_test_{framework}.ipynb shows how to perform a generic test that allows to compare different framework  
- project embeddings to produce a 2d visual representation of the embedding of a given dataset 

## DATASETS
- subsample_data_raw contains data to be annotated
- subsample_damage_crop contains the result of the semi-supervised annotation procedure 
- subsample_damage_stickers contains the stickers to be applied on images to generate synthteic samples
- subsample_data_no_damage contains images of vehicles with no damages to be modified
- subsample_synthetic_dataset contains the generated synthetic dataset
- subsample_synthetic_dataset_encoded contains the renamed synthetic dataset (_cleaned is the cleaned counterpart)
- subsample_dataset contains a real dataset subsample
- subsample_dataset_encoded contains the renamed real dataset (_cleaned is the cleaned counterpart)
- subsample_test_set contains the result of the supervised annotation procedure 
- subsample_dataset_encoded_test contains an encoded test set 
- subsample_inference_claim contains an example of a dataset (divided in claims) that can be used to get inference of similiar images
- subsample_inference_pics contains an example of a dataset (divided in query and gallery) that can be used to get inference of similiar images
