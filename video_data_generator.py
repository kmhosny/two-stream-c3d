# initial implementation by Shervine Amidi
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://github.com/afshinea/keras-data-generator
# =======================================================================
import numpy as np
import keras
import input_data


class VideoDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 list_IDs,
                 labels,
                 work_directory,
                 batch_size=16,
                 dim=(16, 112, 112),
                 n_channels=1,
                 n_classes=10,
                 shuffle=True,
                 num_of_frames=16,
                 crop_size=112):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.work_directory = work_directory
        self.num_of_frames = num_of_frames
        self.crop_size = crop_size
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
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            print(ID)
            X[i, ] = input_data.get_frames_data(
                self.work_directory + ID, self.num_of_frames, self.crop_size)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
