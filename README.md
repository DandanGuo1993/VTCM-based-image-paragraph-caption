# VTCM-based-image-paragraph-caption

image-paragraph-caption
Matching Visual Features to Hierarchical Semantic Topics for Image Paragraph Captioning, published in IJCV 2022.

(1) The implement details for LSTM-based paragraph generation (VTCM-LSTM). This code is based on a github repository :https://github.com/lukemelas/image-paragraph-captioning

Requirements Python 3.5 PyTorch 1.0+ (with torchvision) cider and coco-caption are the tools to evaluate the performance of our model, which can be downloaded from the website [https://github.com/lukemelas/image-paragraph-captioning]

spacy (to tokenize words) h5py (to store features) scikit-image (to process images)

This code is based on a github repository : https://github.com/lukemelas/image-paragraph-captioning, so you can also create a conda environments refer to this repository. And make sure your computer has the platform :java ,visual c++.

you can preprocess the data and we describe the details in 'VCTM-LSTM/scripts/pre-process-data.txt'

##Train just run 'train.py' and you can change the default settings in 'opts_GBN.py'

Note: If you run this code in Windows system , you may need to install visual studio firstly , and change '.so' to '.dll' files in 'PGBN_sampler.py'.

(2) The implement details for Transformer-based paragraph generation (VTCM-Transformer).

Environment Setup
This code is based on a github repository :M²: Meshed-Memory Transformer https://github.com/aimagelab/meshed-memory-transformer you can create a conda environments refer to this repository. And make sure your computer has the platform :java ,visual c++. We also give the requirements in the './requirements.txt'.

you can preprocess the data following the VTCM-LSTM.

Download pretrained M2-model (You can also directly run the model without utilizing the pretrained M2-Transformer, whose performance is only slightly down.)
you can download pretrained models in the following github repository: https://github.com/aimagelab/meshed-memory-transformer

Evaluation
https://github.com/lukemelas/image-paragraph-captioning

##Train just run the following command: python train.py --batch_size 20 --epochs 500 --pretrain_topic_model --features_path your_features_path --annotation_folder your_annotation_folder --save_path your_save_path --logs_folder your_logs_save_path

##Customize your own dataset VCTM-Transformer can be trained on any image-caption datasets. You just need to preprocess the datasets refer to our datasets.


If you find this repo useful to your project, please cite it with following bib:

@article{guo2022matching,
  title={Matching Visual Features to Hierarchical Semantic Topics for Image Paragraph Captioning},
  author={Guo, Dandan and Lu, Ruiying and Chen, Bo and Zeng, Zequn and Zhou, Mingyuan},
  journal={International Journal of Computer Vision},
  pages={1--18},
  year={2022},
  url={https://link.springer.com/article/10.1007/s11263-022-01624-6},
  pdf={https://arxiv.org/pdf/2105.04143.pdf},
  url_arxiv={https://arxiv.org/abs/2105.04143},
  Note = {(the first two authors contributed equally)},
  publisher={Springer}
}

