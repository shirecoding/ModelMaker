import numpy as np

from .utils import class_or_instance_method
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from modelmaker import reverse_dictionary


class ModelInterface(ABC):
    """
    Base Model
    """

    def __init__(self, *args, **kwargs):
        self.setup(*args, **kwargs)

    def __call__(self, x):
        return self.postprocess(self.predict(self.preprocess(x)), deepcopy(x))

    ###########################################################################################
    ## OVERRIDE
    ###########################################################################################

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError("ModelInterface/setup")

    @abstractmethod
    def preprocess(self, x):
        raise NotImplementedError("ModelInterface/preprocess")

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("ModelInterface/predict")

    @abstractmethod
    def postprocess(self, x, orig):
        raise NotImplementedError("ModelInterface/postprocess")

    @abstractmethod
    def get_model(self):
        raise NotImplementedError("ModelInterface/get_model")

    @abstractmethod
    def save_model(self):
        raise NotImplementedError("ModelInterface/save_model")

    @abstractmethod
    def load_model(self):
        raise NotImplementedError("ModelInterface/load_model")

    @abstractmethod
    def fit_model(self):
        raise NotImplementedError("ModelInterface/fit_model")


class ClassificationModelInterface(ModelInterface):
    """
    Classification Model
    """

    ###########################################################################################
    ## OVERRIDE
    ###########################################################################################

    @property
    def labels(self):
        """
        Returns:
            dictionary: maps label to index
        """
        return {}

    ###########################################################################################
    ## UTILS
    ###########################################################################################

    def one_hot(self, label):
        """
        Args:
            label (str): class label
        Returns:
            vector: one-hot numpy vector encoding of label
        """
        return np.eye(len(self.labels))[self.labels[label]]

    def softmax_to_one_hot(self, softmax):
        return np.eye(len(self.labels))[np.argmax(softmax)]

    def one_hot_to_label(self, onehot):
        return reverse_dictionary(self.labels)[np.argmax(onehot)]
