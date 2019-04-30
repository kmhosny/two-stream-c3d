#!/usr/bin/env python

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from video_data_generator import VideoDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg

WORK_DIR = cfg['WORK_DIR']
CLASS_IND = cfg['CLASS_IND']
TEST_SPLIT_FILE = cfg['TEST_SPLIT_FILE']
CROP_SIZE = 112
BATCH_SIZE = 16
NUM_EPOCHS = 100
BATCH_SIZE = 16
num_of_frames = 16


def read_test_ids():
    class_maping = {}
    f_maping = open(CLASS_IND, 'r')
    lines_maping = list(f_maping)
    for i in range(len(lines_maping)):
        line = lines_maping[i].strip('\n').split()
        class_id = int(line[0]) - 1
        class_name = line[1]
        class_maping[class_name] = class_id
    f_maping.close()

    f = open(TEST_SPLIT_FILE, 'r')
    lines = list(f)
    its = range(len(lines))
    IDs = []
    labels = {}
    for it in its:
        line = lines[it].strip('\n')
        dirname = line
        IDs.append(dirname)
        class_name = dirname.split('/')[0]
        labels[dirname] = class_maping[class_name]
    f.close()
    print("Found %i files for total of %i classes." %
          (len(IDs), len(class_maping.keys())))
    return IDs, labels, class_maping


def init_test_generator():
    ids, ground_truth, class_maping = read_test_ids()

    test_datagen = VideoDataGenerator(
        list_IDs=ids,
        labels=ground_truth,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        n_channels=3,
        n_classes=len(class_maping.keys()))

    return test_datagen


def main():
    test_generator = init_test_generator()
    model_weight_filename = './models/c3d_UCF_finetune_weights-99-0.94.h5'
    model_json_filename = './models/c3d_ucf101_finetune_whole_iter_20000_tf.json'

    print("[Info] Reading model architecture...")
    FC_LAYERS = [4096, 4096, 487]
    print(model_weight_filename, model_json_filename)
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)

    print("[Info] Loading model weights -- DONE!")
    model.compile(
        loss='mean_squared_error', optimizer='sgd', metrics=["accuracy"])
    log_dir = "./test_logs"

    board = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='epoch',
        histogram_freq=0)
    callbacks_list = [board]

    result = model.evaluate_generator(
        generator=test_generator,
        steps=10,
        workers=1,
        use_multiprocessing=False,
        verbose=1)
    prediction_classes = []
    for single_prediction in result:
        prediction_classes.append(np.argmax(single_prediction))


    print("result: ", result, model.metrics_names)


if __name__ == '__main__':
    main()
