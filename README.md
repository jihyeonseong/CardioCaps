# CardioCaps: Attention-based Capsule Network for Class-Imbalanced Echocardiogram Classification (IEEE BigComp24)
* This is the author code implements "CardioCaps: Attention-based Capsule Network for Class-Imbalanced Echocardiogram Classification," a paper accepted at IEEE BigComp 2024.
* It builds upon the code of [DR-CapsNet github](https://github.com/tanishqgautam/Capsule-Networks/tree/master) based on PyTorch.
* For further details, please refer to the original [DR-CapsNet](https://arxiv.org/abs/1710.09829) papers.
## Overview
![image](https://github.com/jihyeonseong/CardioCaps/assets/159874470/584143ff-5690-4020-b975-025485df61cc)
Capsule Neural Networks (CapsNets) is a novel architecture that utilizes vector-wise representations formed by multiple neurons. Specifically, the Dynamic Routing CapsNets (DR-CapsNets) employ an affine matrix and dynamic routing mechanism to train capsules and acquire translation-equivariance properties, enhancing its robustness compared to traditional Convolutional Neural Networks (CNNs). Echocardiograms, which capture moving images of the heart, present unique challenges for traditional image classification methods. In this paper, we explore the potential of DR-CapsNets and propose CardioCaps, a novel attention-based DR-CapsNet architecture for class-imbalanced echocardiogram classification. 
* We introduce DR-CapsNets to the challenging problem of echocardiogram diagnosis.
* We propose a new loss function incorporating a weighted margin loss and L2 regularization loss to handle imbalanced classes in echocardiogram datasets.
* We employ an attention mechanism instead of dynamic routing to achieve training efficiency.
* We demonstrate the robustness of CardioCaps through comprehensive comparisons against various baselines.
## Running the codes
### STEP 1. Download the Echocardiogram datsets
* The datasets can be downloaded form the [EchoNet-LVH](https://echonet.github.io/lvh/).
* Create a directory named "data" and store downloaded datasets within it.
### STEP 2. Data-Preprocessing: video to image
```
python data_preprocessing.py
```
### STEP 3. Train the classifiers including CNN, ResNet, U-Net, ViT, and CardioCaps
For traditional baseline models,
```
python main_cnn.py --model=ViT --train --inference
```
and for CardioCaps,
```
python main_dr.py --model=CardioCaps --train --inference
```
finally, run ipynb file for ML baseline classifiers.
### CardioCaps performance
1. Comparison with ML basleines
![image](https://github.com/jihyeonseong/CardioCaps/assets/159874470/483020e0-9f34-4773-b0b7-b1155684c4db)
2. Comparison with DL baselines
![image](https://github.com/jihyeonseong/CardioCaps/assets/159874470/2d1a41c3-3ae9-4e7a-a2c4-20ad92779457)
3. Comparison with advanced CapsNets
![image](https://github.com/jihyeonseong/CardioCaps/assets/159874470/574b9586-c148-40d6-9241-2e66ede8da2a)
## Citation
```
@INPROCEEDINGS {10488274,
author = {H. Han and J. Seong and J. Choi},
booktitle = {2024 IEEE International Conference on Big Data and Smart Computing (BigComp)},
title = {CardioCaps: Attention-Based Capsule Network for Class-Imbalanced Echocardiogram Classification},
year = {2024},
volume = {},
issn = {},
pages = {287-294},
keywords = {weight measurement;training;neurons;routing;sampling methods;loss measurement;robustness},
doi = {10.1109/BigComp60711.2024.00052},
url = {https://doi.ieeecomputersociety.org/10.1109/BigComp60711.2024.00052},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {feb}
}
```
