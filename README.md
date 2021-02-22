# 3D convolution with two-stream convNet for Human Action Recognition
this code is part of submitted thesis under the title 3D convolution with two-stream convNet for Human Action Recognition at The American University in Cairo (AUC). it uses two-streams convolutional network to perform human action recognition, it consists of 2 streams, 1 spatial using ResNet50 and 1 temporal using C3D network.
it uses Tensorflow 1.13.1 and Keras 2.2.4.

This code uses some source code published by https://github.com/hx173149/C3D-tensorflow/ mainly the input_data.py and uses the exported C3D JSON model converted from caffe to tensorflow keras by https://github.com/axon-research/c3d-keras.
## Steps to train on UCF-101:
- clone the repo
- create a new directory 'models' under the source code directory
- download the sports-1M weights to models directory if to use pretrained temporal stream based on C3D https://www.dropbox.com/s/pan3pa6m95c05z7/c3d-sports1M_weights.h5?dl=0
- download UCF-101 data https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
- download splits https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
- unrar the datasets into a directory e.g. UCF-101
- make sure you have ffmpeg installed
- from the source code directory run 
 `./convert_video_to_images.sh ~/datasets/UCF101/ 5`
- it's recommended to use anaconda
> wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
> bash Anaconda2-5.2.0-Linux-x86_64.sh
- create new conda env 
 > conda create --name tf --file packages_without_pip.txt
 > conda activate tf
- install the following
 `pip install cython==0.29.6`
`pip install docutils==0.14`
`pip install scikit-learn==0.20.3`
`pip install sklearn==0.0`
`pip install tensorboard==1.13.1`
`pip install tensorflow==1.13.1`
`pip install tensorflow-estimator==1.13.0`
`pip install keras==2.2.4`
`pip install keras-applications==1.0.7`
`pip install keras-preprocessing==1.0.9`
`pip install lockfile==0.12.2`
`pip install mock==2.0.0`
`pip install opencv-python==4.0.0.21`
`pip install pbr==5.1.3`
`pip install protobuf==3.7.1`
`pip install pybeautifier==0.1.1`
`pip install python-daemon==2.2.3`
`pip install yapf==0.26.0`
- create config.yml with to the keys below
 `WORK_DIR: '{UCF_EXTRACTION_DIR}'`
`CLASS_IND: '{UCF_CLASS_NAME_FILE}'`
`TEST_SPLIT_FILE: '{UCF_TEST_SPLIT_FILE}'`
`TRAIN_SPLIT_FILE: '{UCF_TRAIN_SPLIT_FILE}'`
`ONE_NETWORK_WEIGHTS: '{FINAL_TRAINED_MODEL_WEIGHT_h5_FILE}'`
`PRETRAINED_VIDEO_MODEL: TRUE #USE_C3D_OR_FROM_SCRATCH`
`NUM_OF_FRAMES: 16`
`NUM_OF_CLASSES: 101`
`CROP_SIZE: 64`
`SPORTS_1M_LIST: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_train.txt'`
`SPORTS_1M_LIST_SUBSET: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_train_subset.txt'`
`SPORTS_1M_LIST_SUBSET_SORTED: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_train_subset_sorted.txt'`
`SPORTS_DATASET_DIR: '/home/kmhosny/datasets/SP1M/'`
`SPORTS_DATASET_LABELS: '/home/kmhosny/workspace/sports-1m-dataset/labels.txt'`
`SPORTS_1M_TEST_LIST: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_test.txt'`
`SPORTS_1M_TEST_LIST_SUBSET: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_test_subset.txt'`
`SPORTS_1M_TEST_LIST_SUBSET_SORTED: '/home/kmhosny/workspace/sports-1m-dataset/cross-validation/sports0_test_subset_sorted.txt'`
- run **one_network.py** to start training.
- run **one_network_predict.py** to test your model against test split
- run **one_network_resume.py** to resume training in case of any halting occured during training.
- run **one_network_crop_X.py** to train your model on different crop sizes of the input frame dimension.
