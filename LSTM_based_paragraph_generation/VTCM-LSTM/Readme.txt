#Matching Visual Features to Hierarchical Semantic Topics for Image Paragraph Captioning

Code and dataset for the model VTCM-LSTM in the paper "Matching Visual Features to Hierarchical Semantic Topics for Image Paragraph Captioning", submitted to ICML 2021.

## Environment Setup
This code is based on a github repository :https://github.com/lukemelas/image-paragraph-captioning

Requirements
Python 3.5
PyTorch 1.0+ (with torchvision)
cider and coco-caption are the tools to evaluate the performance of our model, which can be downloaded from the website [https://github.com/lukemelas/image-paragraph-captioning]

spacy (to tokenize words)
h5py (to store features)
scikit-image (to process images)

This code is based on a github repository : https://github.com/lukemelas/image-paragraph-captioning, so you can also create a conda environments refer to this repository. 
And make sure your computer has the platform :java ,visual c++.


you can preprocess the data and we describe the details in 'VCTM-LSTM/scripts/pre-process-data.txt'

##Train 
just run 'train.py' and you can change the default settings in 'opts_GBN.py'

Note:
If you run this code in Windows system , you may need to install visual studio firstly , and change '***.so'  to  '***.dll'  files in  'PGBN_sampler.py'.

