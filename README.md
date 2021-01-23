# ModelMaker

Framework and utilities for creating and validating ML models

## Features

- CLI tool for creation and validation of models
- Choose from predefined templates

## Install the tool

This will install the modelmaker CLI tool

```bash
git clone https://github.com/shirecoding/ModelMaker.git
cd ModelMaker
pip3 install ./
```

alternatively install from pypi

```
pip3 install model-maker
```

## List Templates

```bash
modelmaker templates
```

## Create a New Project

```bash
modelmaker new --project MNISTClassifier --package mnistmodel --template default
```

This will create the python package *mnistmodel* inside directory *MNISTClassifier* (also the main class name)

**Current Templates**

- default (simple classification model using mnist dataset example)
- linear_regression
- text_classification

## Project Structure

- The model is packaged as an importable and installable python library
- *scripts/train.py* used for training the model and exporting it to *saved_models*
- *scripts/test.py* gives an example of how to use the packaged model in production

```bash
src/
    MNISTClassifier/
        mnistmodel/ 	# python model package
        saved_model/ 	# this is where the model is saved after training
            mnistmodel
        scripts/
            train.py 	# training script which imports model package, trains model, saves model to saved_model
            test.py 	# example test script on how to use the model package in production
        setup.py
        README.md
        .gitignore
        ...
```

## Model Framework and Pipeline

- Each model inherits from an abstract class *ModelInterface*
- Each model must override *get_model, fit_model, load_model, save_model*
- Each model must define *preprocess, predict, postprocess*
