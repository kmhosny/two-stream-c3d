#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import c3d_model
import sys
import keras.backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from video_data_generator import VideoDataGenerator
from sklearn.model_selection import train_test_split

dim_ordering = K.image_dim_ordering()
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
    dim_ordering))
backend = dim_ordering

WORK_DIR = '/media/kmhosny/01CFE6D64EF8ED00/datasets/UCF-101-2/'
TRAIN_SPLIT_FILE = '/media/kmhosny/01CFE6D64EF8ED00/datasets/ucfTrainTestlist/trainlist01.txt'
TEST_SPLIT_FILE = '/media/kmhosny/01CFE6D64EF8ED00/datasets/ucfTrainTestlist/testlist01.txt'
CROP_SIZE = 112
BATCH_SIZE = 16
nb_train_samples = 16
nb_validation_samples = 800
NUM_EPOCHS = 1
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
    for layer in base_model.layers:
        layer.trainable = False
    base_model.pop()
    base_model.add(Dense(num_classes, activation='softmax', name='prediction'))
    return base_model


def main():
    train_generator, validation_generator = init_generators()
    model_dir = './models'
    model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
    model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    print("[Info] Reading model architecture...")
    FC_LAYERS = [4096, 4096, 487]
    print(model_weight_filename, model_json_filename)
    model = model_from_json(open(model_json_filename, 'r').read())
    dropout = 0.5

    print("[Info] Loading model weights...")
    model.load_weights(model_weight_filename)
    model = build_finetune_model(model, dropout=dropout, num_classes=101)

    print("[Info] Loading model weights -- DONE!")
    model.compile(
        loss='mean_squared_error', optimizer='sgd', metrics=["accuracy"])
    filepath = "./models/c3d_1M_UCF_weights-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, mode='max')
    callbacks_list = [checkpoint]

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
