import numpy as np
import os
import pandas as pd
import sklearn
import sys

from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(file_path)
project_directory = os.path.dirname(current_directory)
sys.path.insert(0, project_directory)
from rgrmodel.models import SimpleRegression

######################################################################
# train SimpleRegression
######################################################################

# load model in training mode
simple_regression = SimpleRegression(mode='training')
model = simple_regression.get_model()

# load data
iris = load_iris()
df = pd.concat(
    [
        pd.DataFrame(data=iris.data, columns=iris.feature_names),
        pd.DataFrame(data=iris.target, columns=['species'])
    ],
    axis=1
)

# split train test, onehot encode, generate features and target
X = pd.concat(
    [
        df.drop(['species', 'sepal length (cm)'], axis=1),
        pd.get_dummies(df['species'])
    ],
    axis=1
)
y = df['sepal length (cm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=777)

# train model
model.fit(X_train, y_train)

# save model
model_folder = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    ),
    "saved_models"
)
simple_regression.save_model(model, os.path.join(model_folder, 'regression_model'))
