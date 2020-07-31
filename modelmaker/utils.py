import os
import shutil

def delete_directory(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)

def make_directory(directory, delete=False, exist_ok=True):
    if delete:
        delete_directory(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=exist_ok)
    return os.path.abspath(directory)