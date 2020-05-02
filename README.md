# CondenseUNet
Condensely connected architecture for Cardiac Cine MR Image Segmentation and Quantification

This repository contains code to train state-of-the-art cardiac segmentation networks as described in this paper: CondenseUNet: A Memory-Efficient Condensely-Connected Architecture for Bi-ventricular Blood Pool and Myocardium
Segmentation. This architecture is trained on [MICCAI 2017 ACDC Cardiac segmentation challenge.](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)

In this work, we propose a novel memory-efficient Convolutional Neural Network (CNN) architecture as a modification of both **CondenseNet**, as well as DenseNet for ventricular blood-pool segmentation by introducing a bottleneck block and an upsampling path known as CondenseUNet. Our experiments show that the proposed architecture runs on the Automated Cardiac Diagnosis Challenge (ACDC) dataset using half (50%) the memory requirement of DenseNet and one-twelfth (∼ 8%) of the memory requirements of U-Net, while still maintaining excellent accuracy of cardiac segmentation. We validated the framework on the MICCAI STACOM 2017 challenge ACDC dataset featuring one healthy and four pathology groups whose heart images were acquired throughout the cardiac cycle and achieved the mean dice scores of 96.78% (LV blood-pool), 93.46% (RV blood-pool) and 90.1% (LVMyocardium). These results are promising and promote the proposed methods as a competitive tool for cardiac image segmentation and clinical parameter estimation that has the potential to provide fast and accurate results, as needed for pre-procedural planning and / or pre-operative applications.


# Prerequisites
Our code is compatible with python 3.7 or onward.

We depend on some python packages which need to be installed by the user:

* Tensorflow
* tqdm
* SimpleITK
* sklearn
* numpy
* nibabel
* h5py

# Authors 
* S. M. Kamrul Hasan ([sh3190@rit.edu])()
* Cristian A. Linte

# Contents 

* [Data]()
* [Method]()
* [Training]()
* [Results]()
* [How to Run]()



[Data]() \\
For this study, we used the ACDC dataset, which is composed of short-axis cardiac cine-MR images acquired from 100 different patients divided into 5 evenly distributed subgroups according to their cardiac condition: normal- NOR, myocardial infarction- MINF, dilated cardiomyopathy- DCM, hypertrophic cardiomyopathyHCM, and abnormal right ventricle- ARV, available as a part of the STACOM 2017 ACDC challenge.

[Method]() \\
We evaluate CondenseUNet architectures for segmentation as well as clinical parameter estimation. The output of the model is a pixel-by-pixel mask that shows the class of each pixel.

Our proposed L-CO-Net framework substitutes the concept of both standard convolution and group convolution (G-Conv) with learned group-convolution (LGConv). While the standard convolution needs an increased level of computation, i.e. O(Ii x Oo), and concurrently, the pre-defined use of filters in each group convolution restricts its representation capability, these aforementioned problems are mitigated by introducing LG-Conv that learns group convolution dynamically during training through a multi-stage scheme.

![5](https://user-images.githubusercontent.com/42282006/80853422-ccac0100-8bfe-11ea-8c4c-2326c0de7379.png)

[Training]()

<img width="678" alt="Screen Shot 2020-05-01 at 10 56 46 PM" src="https://user-images.githubusercontent.com/42282006/80853458-244a6c80-8bff-11ea-8c52-913b5ff052a6.png">
<img width="643" alt="Screen Shot 2020-05-01 at 10 56 40 PM" src="https://user-images.githubusercontent.com/42282006/80853460-26acc680-8bff-11ea-93bf-7c1533aa062b.png">



[Results]()\\

![Hasan1](https://user-images.githubusercontent.com/42282006/80853394-9b333580-8bfe-11ea-9bae-7227c0431d66.png)
![Hasan3](https://user-images.githubusercontent.com/42282006/80853397-9d958f80-8bfe-11ea-8677-9c557db349d5.png)



If you find this work useful for your publications, please consider citing:

> Hasan, SM Kamrul, and Cristian A. Linte. "CondenseUNet: a memory-efficient condensely-connected architecture for bi-ventricular blood pool and myocardium segmentation." Medical Imaging 2020: Image-Guided Procedures, Robotic Interventions, and Modeling. Vol. 11315. International Society for Optics and Photonics, 2020.'

> Kamrul Hasan, S. M., and Cristian A. Linte. "CondenseUNet: A Memory-Efficient Condensely-Connected Architecture for Bi-ventricular Blood Pool and Myocardium Segmentation." arXiv (2020): arXiv-2004.



