import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from video_data_generator import VideoDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
import random
import numpy as np
from keras import backend as bkend

WORK_DIR = cfg['WORK_DIR']
TEST_SPLIT_FILE = cfg['TEST_SPLIT_FILE']
CLASS_IND = cfg['CLASS_IND']
CROP_SIZE = 112
BATCH_SIZE = 100
NUM_EPOCHS = 500
DIM = (112, 112)


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
    test_datagen = ImageDataGenerator()

    return test_datagen


def build_finetune_model(base_model, dropout, num_classes):

    x = base_model.output
    x = Flatten(name='flatten')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


def predict_intermediate_output(path):
    img_np_array = read_img(path)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    finetune_model = build_finetune_model(
        base_model, dropout=0.5, num_classes=101)

    model_weight_filename = './models/static-resnet-500-0.94.h5'
    print("[Info] Reading model architecture...")
    finetune_model.load_weights(model_weight_filename)

    print("[Info] Loading model weights -- DONE!")
    finetune_model.compile(
        optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])

    #intermediate_model = Model(
    #    input=finetune_model.input,
    #    output=finetune_model.get_layer("flatten").output)
    #intermediate_model.summary()
    intermediate_output = finetune_model.predict(img_np_array)
    bkend.clear_session()
    return intermediate_output


def get_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    finetune_model = build_finetune_model(
        base_model, dropout=0.5, num_classes=101)

    model_weight_filename = './models/static-resnet-500-0.94.h5'
    print("[Info] Reading model architecture...")
    finetune_model.load_weights(model_weight_filename)

    print("[Info] Loading model weights -- DONE!")
    finetune_model.compile(
        optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])
    finetune_model._make_predict_function()
    return finetune_model


def read_img(tid):
    directory = WORK_DIR + "" + tid
    filenames = os.listdir(directory)
    total_files = len(filenames)
    predict_on_file = WORK_DIR + "" + tid + "/" + filenames[random.randint(
        0, total_files - 1)]
    img = Image.open(predict_on_file)
    img = cv2.resize(np.array(img), (CROP_SIZE, CROP_SIZE))
    input = []
    input.append(np.array(img))
    img_np_array = np.array(input)
    return img_np_array


class StaticModel:
    def __init__(self):
        self.model = get_model()
        self.graph = tf.get_default_graph()

    def predict(self, path):
        img_np_array = read_img(path)
        with self.graph.as_default():
            intermediate_output = self.model.predict(img_np_array)

        return intermediate_output


def main():
    test_generator = init_test_generator()
    test_ids, labels, class_maping = read_test_ids()

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    finetune_model = build_finetune_model(
        base_model, dropout=0.5, num_classes=101)

    model_weight_filename = './models/static-resnet-500-0.94.h5'
    print("[Info] Reading model architecture...")
    finetune_model.load_weights(model_weight_filename)

    print("[Info] Loading model weights -- DONE!")
    finetune_model.compile(
        optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])
    prediction_classes = []
    miss = 0
    hit = 0
    total = len(test_ids)
    for tid in test_ids:
        img_np_array = read_img(tid)
        result = finetune_model.predict(img_np_array, verbose=1)
        index = np.argmax(result)
        cl = ""
        for klass, kindex in class_maping.items():
            if index == kindex:
                cl = klass
        if labels[tid] == class_maping[cl]:
            hit = hit + 1
        else:
            miss = miss + 1
        prediction_classes.append([predict_on_file, index, cl])
    acc = (hit * 100) / total
    loss = (miss * 100) / total
    print("result: ", hit, miss, total, acc, loss)


if __name__ == '__main__':
    main()
