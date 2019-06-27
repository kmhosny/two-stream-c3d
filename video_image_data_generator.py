# initial implementation by Shervine Amidi
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://github.com/afshinea/keras-data-generator
# =======================================================================
import numpy as np
import keras
import input_data
import random


class VideoImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 work_directory,
                 batch_size=16,
                 c3d_dim=(16, 112, 112),
                 n_channels=1,
                 n_classes=10,
                 shuffle=True,
                 num_of_frames=16,
                 crop_size=112,
                 static_dim=(112, 112)):
        'Initialization'
        self.c3d_dim = c3d_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.work_directory = work_directory
        self.num_of_frames = num_of_frames
        self.crop_size = crop_size
        self.static_dim = static_dim
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        vX, iX, y = self.__data_generation(list_IDs_temp)

        return [vX, iX], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        vX = np.empty((self.batch_size, *self.c3d_dim))
        iX = np.empty((self.batch_size, *self.static_dim))

        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            frame_data = input_data.get_frames_data(
                self.work_directory + ID, self.num_of_frames, self.crop_size)
            image_index = random.randint(0, len(frame_data) - 1)
            im = frame_data[image_index]
            vX[i, ] = frame_data
            iX[i, ] = im
            # Store class
            y[i] = self.labels[ID]

        return vX, iX, keras.utils.to_categorical(
            y, num_classes=self.n_classes)
