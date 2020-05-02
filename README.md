# CondenseUNet
Condensely connected architecture for Cardiac Cine MR Image Segmentation and Quantification

This repository contains code to train state-of-the-art cardiac segmentation networks as described in this paper: CondenseUNet: A Memory-Efficient Condensely-Connected Architecture for Bi-ventricular Blood Pool and Myocardium
Segmentation. This architecture is trained on [MICCAI 2017 ACDC Cardiac segmentation challenge.](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)

In this work, we propose a novel memory-efficient Convolutional Neural Network (CNN) architecture as a modification of both **CondenseNet**, as well as DenseNet for ventricular blood-pool segmentation by introducing a bottleneck block and an upsampling path known as CondenseUNet. Our experiments show that the proposed architecture runs on the Automated Cardiac Diagnosis Challenge (ACDC) dataset using half (50%) the memory requirement of DenseNet and one-twelfth (∼ 8%) of the memory requirements of U-Net, while still maintaining excellent accuracy of cardiac segmentation. We validated the framework on the MICCAI STACOM 2017 challenge ACDC dataset featuring one healthy and four pathology groups whose heart images were acquired throughout the cardiac cycle and achieved the mean dice scores of 96.78% (LV blood-pool), 93.46% (RV blood-pool) and 90.1% (LVMyocardium). These results are promising and promote the proposed methods as a competitive tool for cardiac image segmentation and clinical parameter estimation that has the potential to provide fast and accurate results, as needed for pre-procedural planning and / or pre-operative applications.



