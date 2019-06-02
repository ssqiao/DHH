# DHH
I'm very sorry that I am a little busy recently. I'll update the readme to give an introduction about the usage of the codes a few days later (no more than a week). 
Caffe implementation of our work entitled as "Deep Heterogeneous Hashing for Face Video Retrieval" (TIP under review). For research use only, commercial use is not allowed.

# Prerequisites
Linux 14.04 (We simply tried it on 16.06 or high version but failed due to MKL issue)

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

BLAS lib: Intel MKL V2017.1.132

# Modification on Caffe

# Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions.

# Datasets
We use YTC, PB and a subset with 200 subjects of UMDFaces dataset in our experiments. We have preprocessed these three datasets. You can download them here. As for COCO dataset, we use COCO 2014, which can be downloaded here (BaiduCloud drive). And in the future, we will provide a download link on google drive. After downloading, you need to covert them to the LMDB format which is a more efficient data storage technique.

For video modality, you 

For image modality, you 

# Training
We place the prototxt files . First, you need to download the AlexNet pre-trained model on ImageNet from here and move it to ./models/bvlc_reference_caffenet. Then, you can train the model for each dataset using the followling command

# Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset using the followling command.

We provide some trained models for each dataset for each code length in our experiment for evaluation. You can download them here if you want to use them.

# Contact
If you have any problem about our code, feel free to contact shishi.qiao@vipl.ict.ac.cn or describe your problem in Issues.
