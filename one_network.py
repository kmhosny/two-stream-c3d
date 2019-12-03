import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import keras.backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
from keras.optimizers import SGD
import numpy as np
from feature_data_generator import FeatureDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg
from static_image_predict import get_model as get_static_model
from video_image_data_generator import VideoImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

WORK_DIR = cfg['WORK_DIR']
TEST_SPLIT_FILE = cfg['TEST_SPLIT_FILE']
TRAIN_SPLIT_FILE = cfg['TRAIN_SPLIT_FILE']
NUM_OF_CLASSES = cfg['NUM_OF_CLASSES']
BATCH_SIZE = cfg['NUM_OF_FRAMES']
NUM_EPOCHS = 500
CROP_SIZE = 112
C3D_INPUT_SHAPE = (BATCH_SIZE, 112, 112, 3)
STATIC_INPUT_SHAPE = (112, 112, 3)
MODEL_JSON_FILENAME = './models/sports1M_weights_tf_notop.json'
VIDEO_MODEL_TOP = './models/sports1M_weights_tf.json'
MODEL_WEIGHT_FILENAME = './models/sports1M_weights_tf.h5'
PRETRAINED_VIDEO_MODEL = cfg['PRETRAINED_VIDEO_MODEL']


def avg(ve1, ve2):
    return (ve1 + ve2) * 0.5


def read_file_ids(filename):
    f = open(filename, 'r')
    lines = list(f)
    its = range(len(lines))
    IDs = []
    labels = {}
    for it in its:
        line = lines[it].strip('\n').split()
        dirname = line[:-1]
        dirname = " ".join(dirname)
        label = line[-1:][0]
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

    train_datagen = VideoImageDataGenerator(
        list_IDs=train_ids,
        labels=train_labels,
        c3d_dim=C3D_INPUT_SHAPE,
        crop_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        n_channels=3,
        n_classes=len(set(train_labels.values())),
        static_dim=STATIC_INPUT_SHAPE)

    validation_datagen = VideoImageDataGenerator(
        list_IDs=validation_ids,
        labels=validation_labels,
        crop_size=CROP_SIZE,
        c3d_dim=C3D_INPUT_SHAPE,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        n_channels=3,
        n_classes=len(set(train_labels.values())),
        static_dim=STATIC_INPUT_SHAPE)

    return train_datagen, validation_datagen


def build_static_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    x = base_model.output
    x = Flatten(name='flatten')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model


def build_video_model():
    c3d_model = ''
    if PRETRAINED_VIDEO_MODEL:
        model = model_from_json(open(VIDEO_MODEL_TOP, 'r').read())
        model.load_weights(MODEL_WEIGHT_FILENAME)
        c3d_model = Sequential()
        for layer in model.layers[:-1]:
            c3d_model.add(layer)
    else:
        c3d_model = model_from_json(open(MODEL_JSON_FILENAME, 'r').read())
    return c3d_model


def deep_model():
    video_input = Input(shape=C3D_INPUT_SHAPE)
    image_input = Input(shape=STATIC_INPUT_SHAPE)
    static_model = build_static_model()
    c3d_model = build_video_model()
    encoded_c3d = c3d_model(video_input)
    encoded_static = static_model(image_input)
    merged = concatenate([encoded_c3d, encoded_static])
    merge_model = Dense(NUM_OF_CLASSES, activation='softmax')(merged)
    model = Model(inputs=[video_input, image_input], outputs=merge_model)
    sgd = SGD(nesterov=True)
    model.compile(
        loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def merge_streams_output(static_stream, c3d_stream, technique):
    func = merge_technique[technique]
    return func(static_stream, c3d_stream)


vec_avg = np.vectorize(avg)
merge_technique = {0: vec_avg}


def main():
    model = deep_model()
    train_generator, validation_generator = init_generators()
    filepath = "./models/one_network_scratch-"+str(NUM_OF_CLASSES)+"-"+str(BATCH_SIZE)+"frs{val_acc:.2f}.h5"
    log_dir = "./one_network_logs/{}/500-pretrained-{}frs/".format(NUM_OF_CLASSES, BATCH_SIZE)
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, mode='max')
    board = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='epoch',
        histogram_freq=0)
    callbacks_list = [checkpoint, board]

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCHS,
        workers=1,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=callbacks_list)


if __name__ == '__main__':
    main()
