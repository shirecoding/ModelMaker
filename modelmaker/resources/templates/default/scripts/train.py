import os
import sys

import numpy as np
from tensorflow import keras

file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(file_path)
project_directory = os.path.dirname(current_directory)
sys.path.insert(0, project_directory)
from {{package_name}}.models import {{ project_name }}

######################################################################
# train {{ project_name }}
######################################################################

# load model in training mode
model = {{ project_name }}()

class DataGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, mode='train'):
        # load mnist dataset
        train, test = keras.datasets.mnist.load_data(path="mnist.npz")

        if mode == 'train':
            self.xs, self.ys = train
        else:
            self.xs, self.ys = test

        self.total = self.xs.shape[0]
        self.batch_size = batch_size


    def __len__(self):
        return self.total // self.batch_size

    def __getitem__(self, idx):

        start = idx * self.batch_size
        end = self.batch_size + idx * self.batch_size

        if self.batch_size + idx * self.batch_size  > self.total:
            end = self.total

        xs = np.asarray([ model.preprocess(x) for x in self.xs[start:end,:,:] ])
        ys = np.asarray([ model.one_hot(y) for y in self.ys[start:end] ])

        return xs, ys

# training parameters
batch_size = 32
epochs = 5

# generate training data
train_data = DataGenerator(batch_size, mode='train')
val_data = DataGenerator(batch_size, mode='test')

# train model
model.fit_model(train_data, val_data, epochs=epochs)

# save model
model_folder = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    ),
    "saved_models"
)
model.save_model(os.path.join(model_folder, '{{ package_name }}'))
