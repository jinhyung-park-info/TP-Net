import tensorflow as tf
import numpy as np
import cv2


class PredictionModelDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, input_sequences_filepaths, output_sequences_filepaths, batch_size):
        self.input_filenames = input_sequences_filepaths
        self.output_filenames = output_sequences_filepaths
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.input_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.input_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.output_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = [pointset for file_name in batch_x for pointset in np.load(file_name)]
        y = [pointset for file_name in batch_y for pointset in np.load(file_name)]

        return x, y


class EncoderDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, batch_size, labels=None):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.labels is not None:
            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([cv2.imread(str(file_name), 0) for file_name in batch_x]) / 255.0, np.array(batch_y)
        else:
            return np.array([cv2.imread(str(file_name), 0) for file_name in batch_x]) / 255.0


class DecoderDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, labels, batch_size, image_filenames=None):
        self.labels = labels
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.labels) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.image_filenames is not None:
            batch_y = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array(batch_x) - 0.5, np.array([cv2.imread(str(file_name), 0) for file_name in batch_y]) / 255.0
        else:
            return np.array(batch_x) - 0.5
