import os
import sys
import numpy as np
from tensorflow import keras
file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(file_path)
project_directory = os.path.dirname(current_directory)
sys.path.insert(0, project_directory)
from {{ package_name }}.models import TextClassification

# load model in development mode
model_path = os.path.join(project_directory, 'saved_models', 'imdb_sentiment_model')
text_classifier = TextClassification(mode='development', model_path=model_path)

sample_text = [
	'The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.'
]

print(text_classifier(sample_text))