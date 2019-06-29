import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import keras.backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
import numpy as np
from feature_data_generator import FeatureDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg
from static_image_predict import get_model as get_static_model
from video_image_data_generator import VideoImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

WORK_DIR = cfg['WORK_DIR']
CLASS_IND = cfg['CLASS_IND']
TEST_SPLIT_FILE = cfg['TEST_SPLIT_FILE']
TRAIN_SPLIT_FILE = cfg['TRAIN_SPLIT_FILE']
MODEL_WEIGHTS_FILE = cfg['ONE_NETWORK_WEIGHTS']
NUM_OF_CLASSES = 101
BATCH_SIZE = 16
NUM_EPOCHS = 500
CROP_SIZE = 112
C3D_INPUT_SHAPE = (16, 112, 112, 3)
STATIC_INPUT_SHAPE = (112, 112, 3)
MODEL_JSON_FILENAME = './models/c3d_ucf101_finetune_whole_iter_20000_tf_notop.json'


def avg(ve1, ve2):
    return (ve1 + ve2) * 0.5


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


def build_static_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    x = base_model.output
    x = Flatten(name='flatten')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def deep_model():
    video_input = Input(shape=C3D_INPUT_SHAPE)
    image_input = Input(shape=STATIC_INPUT_SHAPE)
    c3d_model = model_from_json(open(MODEL_JSON_FILENAME, 'r').read())
    static_model = build_static_model()
    encoded_c3d = c3d_model(video_input)
    encoded_static = static_model(image_input)
    merged = concatenate([encoded_c3d, encoded_static])
    merge_model = Dense(NUM_OF_CLASSES, activation='softmax')(merged)
    model = Model(inputs=[video_input, image_input], outputs=merge_model)

    return model


def merge_streams_output(static_stream, c3d_stream, technique):
    func = merge_technique[technique]
    return func(static_stream, c3d_stream)


vec_avg = np.vectorize(avg)
merge_technique = {0: vec_avg}


def main():
    model = deep_model()
    test_generator = init_test_generator()
    model.load_weights(ONE_NETWORK_WEIGHTS)
    model.compile(
        loss='mean_squared_error', optimizer='nadam', metrics=['accuracy'])

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
