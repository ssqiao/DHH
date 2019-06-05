# DHH
Caffe implementation of our work entitled as "Deep Heterogeneous Hashing for Face Video Retrieval" (TIP2019 under review). For research use only, commercial use is not allowed.

# Prerequisites
Linux 14.04 (We simply tried it on 16.06 or high version but failed due to MKL issue)

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

BLAS lib: Intel MKL V2017.1.132

# Modification on Caffe
- Add covert_imageset_set in the tools which converts video clips into lmdb format
- Add extract_features_binary in the tools which extracts the outputs of one layer of a trained model into binary file
- Modified db, db_leveldb, db_lmdb, data_reader, data_layer which deal with the image and video data in lmdb format
- Modified math_functions in the utils which now supports the svd and more matrix operations with the help of MKL BLAS
- Add sub_mean_layer, covlogm_layer, temporal_pooling_layer which handle the video modeling procedure for face videos
- Add triplet_rank_loss and other metric learning loss which are used for hashing supervision
- Modified caffe.proto to support corresponding modifications listed above

# Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

# Datasets
We use YTC, PB and a subset with 200 subjects of UMDFaces dataset in our experiments. We have preprocessed these three datasets. You can download them here. As for COCO dataset, we use COCO 2014, which can be downloaded here (BaiduCloud drive). And in the future, we will provide a download link on google drive. After downloading, you need to covert them to the LMDB format which is a more efficient data storage technique.

For video modality, you 

For image modality, you 

# Training
We place the prototxt files in the prototxt folder. First, you need to download the pre-trained model from here and move it to ./models/. Then, you can train the model for each dataset using the followling command

# Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset using the followling command.

We provide some trained models for each dataset for each code length in our experiment for evaluation. You can download them here if you want to use them.

# Contact
If you have any problem about our code, feel free to contact shishi.qiao@vipl.ict.ac.cn or describe your problem in Issues.
