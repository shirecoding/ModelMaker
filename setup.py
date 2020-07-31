from distutils.core import setup
from setuptools import find_packages
from modelmaker import __version__

setup(
    name='modelmaker',
    version=__version__,
    author='shirecoding',
    scripts=['bin/modelmaker'],
    install_requires=[],
    url='https://github.com/shirecoding/ModelMaker',
    long_description=open('README.md').read(),
    zip_safe=False,
    include_package_data=True,
    packages=find_packages() + [
        'modelmaker.resources'
    ],
    package_data={
        'modelmaker.resources': [
            'project_template/*',
            'project_template/**/*',
            'templates/*',
            'templates/**/*',
        ]
    }
)