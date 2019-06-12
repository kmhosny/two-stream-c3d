import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
import numpy as np
from feature_data_generator import FeatureDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg

WORK_DIR = cfg['WORK_DIR']
TEST_SPLIT_FILE = cfg['TEST_SPLIT_FILE']
TRAIN_SPLIT_FILE = cfg['TRAIN_SPLIT_FILE']
NUM_OF_CLASSES = 101
BATCH_SIZE = 16
NUM_EPOCHS = 100
CROP_SIZE = 112


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

    train_datagen = FeatureDataGenerator(
        list_IDs=train_ids,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        fusion_method=merge_streams_output,
        fusion_technique=0,
        n_channels=1,
        n_classes=len(set(train_labels.values())))

    validation_datagen = FeatureDataGenerator(
        list_IDs=validation_ids,
        labels=validation_labels,
        batch_size=BATCH_SIZE,
        work_directory=WORK_DIR,
        fusion_method=merge_streams_output,
        fusion_technique=0,
        n_channels=1,
        n_classes=len(set(train_labels.values())))

    return train_datagen, validation_datagen


def deep_model():
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=NUM_OF_CLASSES))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES, activation='softmax'))
    model.compile(
        loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model


def merge_streams_output(static_stream, c3d_stream, technique):
    func = merge_technique[technique]
    print("static stream shape", static_stream.shape)
    print("c3d stream shape", c3d_stream.shape)
    return func(static_stream, c3d_stream)


vec_avg = np.vectorize(avg)
merge_technique = {0: vec_avg}


def main():
    train_generator, validation_generator = init_generators()
    model = deep_model()
    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=NUM_EPOCHS,
        workers=1)


if __name__ == '__main__':
    main()
