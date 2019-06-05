# DHH
Caffe implementation of our work entitled as "Deep Heterogeneous Hashing for Face Video Retrieval" (TIP2019 under review). For research use only, commercial use is not allowed.

# Prerequisites
Linux 14.04 (We simply tried it on 16.06 or high version but failed due to MKL issue)

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

BLAS lib: Intel MKL V2017.1.132

# Modification on Caffe
- Add convert_imageset_set in the tools which converts video clips into lmdb format
- Add extract_features_binary in the tools which extracts the outputs of one layer of a trained model into binary file
- Modified db, db_leveldb, db_lmdb, data_reader, data_layer which deal with the image and video data in lmdb format during training and testing 
- Modified math_functions in the utils which now supports the svd and more matrix operations with the help of MKL BLAS
- Add sub_mean_layer, covlogm_layer, temporal_pooling_layer which handle the video modeling procedure for face videos
- Add triplet_rank_loss and other metric learning loss which are used for hashing supervision
- Modified caffe.proto to support corresponding modifications listed above

# Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

# Datasets
We use YTC, PB and a subset containing 200 subjects of UMDFaces dataset in our experiments. We have preprocessed these three datasets and provided both the raw images and the converted lmdb files for direct training and testing. You can download them [here](https://pan.baidu.com/s/1lVWcqujE8kMUqQLTIAEBzw) using the extracted codes：zic7 (BaiduCloud drive). And in the future, we will provide a download link on google drive.

After downloading, you can directly use the lmdb files for training and testing DHH. Also you can convert the raw images together with split txt files to the LMDB format as we have provided for you.
For video modality, you can use the following command for PB dataset as an example to convert the video clips:
```
./build/tools/convert_imageset_set --resize_height=64 --resize_width=64 path/to/orig_imgs  path/to/train_test_fold  /path/to/list txt file  path/to/saved lmdb file
```
For image modality, you can use the following command for PB dataset as an example to convert the still images:
```
./build/tools/convert_imageset --resize_height=64 --resize_width=64 path/to/orig_imgs  path/to/list txt file path/to/saved lmdb file 
```

# Training
We place the solver and net prototxt files in the prototxt folder. First, you need to download the pre-trained classification model for initilizing our method from [here](https://pan.baidu.com/s/1lVWcqujE8kMUqQLTIAEBzw) using the extracted codes：zic7 (BaiduCloud drive) and move it to ./models/. Then, you need to modify the corresponding paths in the solver and net prototxt files. Finaly, you can train the model for each dataset using the followling command:
```
./build/tools/caffe train --solver ./prototxt/PB/solver_dhh_12.prototxt --weights ./models/PB/pb_classification_iter_5000.caffemodel
```

After this step, you can further improve the cross-modality retrieval performance by finetuning the model with only the inter-space triplet loss:
```
./build/tools/caffe train --solver ./prototxt/PB/solver_dhh_cross_12.prototxt --weights path/to/your trained dhh model in above step
```

# Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset with the provided evaluation scripts in matlab. First, you need to extract the binary codes and labels of videos and images using the following command:
```
./build/tools/extract_features_binary path/to/trained DHH models ./prototxt/PB/train_val_dhh_12.txt  ip1 (hash layer output of videos) path/to/save binary file

./build/tools/extract_features_binary path/to/trained DHH models ./prototxt/PB/train_val_dhh_12.txt merge_label (labels of videos) path/to/save binary file 
```

And then you can modify the binary files path in matlab script, and run it in matlab environment to obtain the mAP result:
```
Evaluate_DHH.m
```

We provide our trained DHH models for each dataset under each code length in our experiment for evaluation. You can download them [here](https://pan.baidu.com/s/1lVWcqujE8kMUqQLTIAEBzw) using the extracted codes：zic7 (BaiduCloud drive) if you want to use them.

# Contact
If you have any problem about our code, feel free to contact shishi.qiao@vipl.ict.ac.cn or describe your problem in Issues.
