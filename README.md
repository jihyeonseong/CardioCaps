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
