## Crack segmentation
# Background
* This repository is for crack segmentation using Deep learning.
* It provides semantic segmentation model based on Swin-Transformer.
* Swin-Transformer is SOTA(State of Art) in vision field previously applicated on NLP model.
* Also, some useful functions are available such as sliding inference, image segmentation on raw resolution image

# Description
* Train codes are located in the following directory.
  './main/train/'
  
* data preparation codes are located in the following diectory.
  './main/data_preparation/'
  
* computation of performance index using sliding inference codes are located in the following directory.
  './main/inference/'
  
* util functions(inference tool, sliding segmentation etc..) are located in the following directory.
  './utils/'
  
# Customization
* some train configuration and hyperparameter can be modified in the following code.
  './configs/swin/kepco_crack.py
* sample dataset is provided in the following directory. 5 datasets are available and used as pre-trained dataset.
* Crack images are from building, bridges, various infrastructures.
* User should add dataset as follow.
  './data/
      dataset1/
          image/*.jpg .....
          label/*.png .....
