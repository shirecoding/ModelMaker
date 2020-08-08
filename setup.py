from distutils.core import setup
from setuptools import find_packages
from modelmaker import __version__

setup(
    name='modelmaker',
    version=__version__,
    author='shirecoding',
    scripts=['bin/modelmaker'],
    install_requires=[
        'Jinja2',
        'opencv-python',
        'numpy'
    ],
    url='https://github.com/shirecoding/ModelMaker',
    download_url='https://github.com/shirecoding/ModelMaker/archive/0.0.1.tar.gz',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    packages=find_packages() + [
        'modelmaker.resources'
    ],
    package_data={
        'modelmaker.resources': [
            'templates/*',
            'templates/**/*',
            'templates/**/**/*',
            'templates/**/**/**/*',
            'templates/**/**/**/**/*',
        ]
    }
)