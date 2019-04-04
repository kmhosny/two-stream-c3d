import keras.backend as K
from keras.models import Model
import numpy as np

NUM_OF_CLASSES = 101
BATCH_SIZE = 16
NUM_EPOCHS = 100
TRAIN_SPLIT_FILE = [
    './datasets/UCF-101/trainlist01-static-features.txt',
    './datasets/UCF-101/trainlist01-c3d-features.txt'
]
TEST_SPLIT_FILE = [
    './datasets/UCF-101/testlist01-static-features.txt',
    './datasets/UCF-101/testlist01-c3d-features.txt'
]


def avg(ve1, ve2):
    return (ve1 + ve2) * 0.5


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
    return func(static_stream, c3d_stream)


vec_avg = np.vectorize(avg)
merge_technique = {0: vec_avg}


def main():
    a = np.array([2, 3, 4])
    b = np.array([4, 5, 6])
    print(merge_streams_output(a, b, 0))
    model = deep_model()
    model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
