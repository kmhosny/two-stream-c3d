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
TRAIN_SPLIT_FILE = '/home/kmhosny/datasets/ucfTrainTestlist/trainlist01.txt'
TEST_SPLIT_FILE = '/home/kmhosny/datasets/ucfTrainTestlist/testlist01.txt'
CROP_SIZE = 112
BATCH_SIZE = 16
NUM_EPOCHS = 1000
BATCH_SIZE = 16
num_of_frames = 16


def read_file_ids(filename):
    f = open(filename, 'r')
    lines = list(f)
    its = range(len(lines))
    IDs = []
    labels = {}
    for it in its:
        line = lines[it].strip('\n').split()
        dirname = line[0]
        label = line[1]
        IDs.append(dirname)
        labels[dirname] = int(label) - 1
    f.close()
    print("Found %i files belonging to %i classes" %
          (len(IDs), len(set(labels.values()))))
    return IDs, labels


def split_train_validation(IDs, labels):
    train_ids, validation_ids, _, _ = train_test_split(
        IDs,
        list(labels.keys()),
        test_size=0.20,
        random_state=len(set(labels.values())))

    train_labels = {}
    for label in train_ids:
        train_labels[label] = labels[label]

    validation_labels = {}
    for label in validation_ids:
        validation_labels[label] = labels[label]

    return train_ids, train_labels, validation_ids, validation_labels


def init_generators():
    ids, labels = read_file_ids(TRAIN_SPLIT_FILE)
    train_ids, train_labels, validation_ids, validation_labels = split_train_validation(
        ids, labels)

    train_datagen = VideoDataGenerator(
        list_IDs=train_ids,
        labels=train_labels,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        n_channels=3,
        n_classes=len(set(train_labels.values())))

    validation_datagen = VideoDataGenerator(
        list_IDs=validation_ids,
        labels=validation_labels,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        n_channels=3,
        n_classes=len(set(train_labels.values())))
    return train_datagen, validation_datagen


def build_finetune_model(base_model, dropout, num_classes):
    total_layers_len = len(base_model.layers)
    for layer in base_model.layers[:total_layers_len - 1]:
        layer.trainable = False


#    base_model.pop()
#   base_model.add(Dense(num_classes, activation='softmax', name='prediction'))
    return base_model


def main():
    train_generator, validation_generator = init_generators()
    model_json_filename = './models/c3d_ucf101_finetune_whole_iter_20000_tf.json'

    print("[Info] Reading model architecture...")
    print(model_json_filename)
    model = model_from_json(open(model_json_filename, 'r').read())

    print("[Info] Loading model weights -- DONE!")
    model.compile(
        loss='mean_squared_error', optimizer='sgd', metrics=["accuracy"])
    filepath = "./models/c3d_UCF_scratch_weights-{epoch:02d}-{val_acc:.2f}.h5"
    log_dir = "./logs"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, mode='max')
    board = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='epoch',
        histogram_freq=0)
    callbacks_list = [checkpoint, board]

    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCHS,
        workers=2,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=callbacks_list)


if __name__ == '__main__':
    main()
