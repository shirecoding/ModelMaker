import numpy as np
import os
import pandas as pd
import sys

from sklearn.datasets import load_iris

file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(file_path)
project_directory = os.path.dirname(current_directory)
sys.path.insert(0, project_directory)
from {{ package_name }}.models import SimpleRegression

# load model in development mode
model_path = os.path.join(project_directory, 'saved_models', 'regression_model')
regression_model = SimpleRegression(mode='development', model_path=model_path)

# get input data
iris = load_iris()
df = pd.concat(
    [
        pd.DataFrame(data=iris.data, columns=iris.feature_names),
        pd.DataFrame(data=iris.target, columns=['species'])
    ],
    axis=1
)
X = pd.concat(
    [
        df.drop(['species', 'sepal length (cm)'], axis=1),
        pd.get_dummies(df['species'])
    ],
    axis=1
)
Y = df['sepal length (cm)']

# predict
for x, y in zip(X[1:10].values, Y[1:10]):
	print(f"x: {x}, y_hat: {regression_model.predict([x])}, y: {y}")


