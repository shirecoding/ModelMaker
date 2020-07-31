from distutils.core import setup
from setuptools import find_packages
from modelmaker import __version__

setup(
    name='modelmaker',
    version=__version__,
    author='shirecoding',
    packages=find_packages(),
    install_requires=[],
    url='https://github.com/shirecoding/ModelMaker',
    long_description=open('README.md').read(),
    zip_safe=False,
    scripts=['bin/modelmaker']
)