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
from video_data_generator import VideoDataGenerator
from sklearn.model_selection import train_test_split
from configuration import cfg
import matplotlib.pyplot as plt

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


def main():
    test_generator = init_test_generator()
    test_ids, labels, class_maping = read_test_ids()

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))
    finetune_model = build_finetune_model(
        base_model, dropout=0.5, num_classes=101)

    model_weight_filename = './models/static_model_resnet-99-0.94.h5'
    print("[Info] Reading model architecture...")
    finetune_model.load_weights(model_weight_filename)

    print("[Info] Loading model weights -- DONE!")

    for tid in test_ids:
        directory = WORK_DIR + "" + tid
        test_data = test_generator.flow_from_directory(
            directory,
            target_size=DIM,
            batch_size=BATCH_SIZE,
            class_mode="categorical")
        result = finetune_model.evaluate_generator(
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
