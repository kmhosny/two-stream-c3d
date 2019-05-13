import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
TRAIN_SPLIT_FILE = cfg['TRAIN_SPLIT_FILE']
CROP_SIZE = 112
BATCH_SIZE = 16
NUM_EPOCHS = 500
DIM = (112, 112)


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
    # ids, labels = read_file_ids(TRAIN_SPLIT_FILE)
    # train_ids, train_labels, validation_ids, validation_labels = split_train_validation(
    #     ids, labels)
    #
    # train_datagen = VideoDataGenerator(
    #     list_IDs=train_ids,
    #     labels=train_labels,
    #     crop_size=CROP_SIZE,
    #     batch_size=BATCH_SIZE,
    #     dim=DIM,
    #     work_directory=WORK_DIR,
    #     n_channels=3,
    #     num_of_frames=1,
    #     n_classes=len(set(train_labels.values())))
    #
    # validation_datagen = VideoDataGenerator(
    #     list_IDs=validation_ids,
    #     labels=validation_labels,
    #     crop_size=CROP_SIZE,
    #     batch_size=BATCH_SIZE,
    #     dim=DIM,
    #     work_directory=WORK_DIR,
    #     n_channels=3,
    #     num_of_frames=1,
    #     n_classes=len(set(train_labels.values())))
    train_datagen = ImageDataGenerator(validation_split=0.2)
    #return train_datagen, validation_datagen
    return train_datagen


def build_finetune_model(base_model, dropout, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')


def main():
    #train_generator, validation_generator = init_generators()
    train_generator = init_generators()
    train_data = train_generator.flow_from_directory(
        WORK_DIR,
        target_size=DIM,
        batch_size=BATCH_SIZE,
        class_mode="categorical")
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(CROP_SIZE, CROP_SIZE, 3))

    finetune_model = build_finetune_model(
        base_model, dropout=0.5, num_classes=101)

    finetune_model.compile(
        optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])

    filepath = "./models/static-resnet-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, mode='max')
    log_dir = "./logs/static_model/"
    board = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='epoch',
        histogram_freq=0)
    callbacks_list = [checkpoint, board]

    history = finetune_model.fit_generator(
        train_data,
        epochs=NUM_EPOCHS,
        steps_per_epoch=500,
        workers=2,
        shuffle=True,
        callbacks=callbacks_list)

    plot_training(history)


if __name__ == '__main__':
    main()
