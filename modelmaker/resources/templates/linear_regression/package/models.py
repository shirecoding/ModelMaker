import numpy as np
from modelmaker import ModelInterface
from sklearn import linear_model
import pickle

class SimpleRegression(ModelInterface):

    def setup(self, mode='production', model_path=None):
        self.mode = mode
        if mode == 'development':
            self.model = self.load_model(model_path)
        elif mode == 'production':
            raise Exception('production mode not implemented')
        elif mode == 'training':
            pass
        else:
            raise Exception('invalid mode')

    def get_model(self):
        return linear_model.LinearRegression()

    def save_model(self, model, path):
        with open(path, 'wb') as f:
            return pickle.dump(model, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def preprocess(self, x):
        return x

    def predict(self, x):
        if self.mode == "production":
            raise Exception('production mode not implemented')
        elif self.mode == 'development':
            return self.model.predict(x)
        else:
            raise Exception('invalid mode')

    def postprocess(self, x, orig):
        return x