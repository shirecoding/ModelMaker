import numpy as np
import tensorflow as tf

from .utils import normalize
from .utils import rescale2d
from modelmaker import ClassificationModelInterface
from tensorflow import keras

class TextClassification(ClassificationModelInterface):

    def setup(self, mode='production', model_path=None):
        self.mode = mode

        if mode == 'development':
            self.model = keras.models.load_model(model_path)
        elif mode == 'production':
            raise Exception('production mode not implemented')
        elif mode == 'training':
            pass
        else:
            raise Exception('invalid mode')

    @property
    def labels(self):
        return {
            0: 0,
            1: 1
        }

    def get_model(self, encoder):
        """ Get Model
        Args:
            encoder (keras.layers.experimental.preprocessing.TextVectorization): text encoder for tokenizing text

        Returns:
            keras.Model: model
        """

        model = keras.Sequential([
            encoder,
            keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(64)
            ),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ])

        return model

    def preprocess(self, xs):
        """
        Args:
            xs (list[str]): list of text strings
        """
        return np.array([*xs])

    def predict(self, x):
        """
        Args:
            x (np.array): input to model
        
        Returns:
            np.array: model scores
        """
        if self.mode == "production":
            raise Exception('production mode not implemented')
        elif self.mode == 'development':
            return self.model.predict(x)
        else:
            raise Exception('invalid mode')

    def postprocess(self, xs, orig):
        return np.ravel(xs)
