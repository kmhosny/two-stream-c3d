#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

WORK_DIR = '/home/kmhosny/datasets/UCF-101/'
CLASS_IND = '/home/kmhosny/datasets/ucfTrainTestlist/classInd.txt'
TEST_SPLIT_FILE = '/home/kmhosny/datasets/ucfTrainTestlist/testlist01.txt'
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
        dirname = line[0]
        IDs.append(dirname)
        class_name = dirname.split('//')[0]
        labels[dirname] = class_maping[class_name]
    f.close()
    print("Found %i files for total of %i classes." %
          (len(IDs), len(class_maping.keys())))
    return IDs, labels, class_maping


def init_test_generator():
    ids, ground_truth, class_maping = read_file_ids()

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
    train_generator, validation_generator = init_generators()
    model_weight_filename = './models/c3d_1M_UCF_weights-10-0.89.h5'
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
    filepath = "./models/c3d_UCF_finetune_weights-{epoch:02d}-{val_acc:.2f}.h5"
    log_dir = "./test_logs"

    board = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='epoch',
        histogram_freq=0)
    callbacks_list = [board]

    history = model.evaluate_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCHS,
        workers=2,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=callbacks_list)


if __name__ == '__main__':
    main()
