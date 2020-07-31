from distutils.core import setup
from package import __version__

setup(
    name='package',
    version=__version__,
    author='',
    install_requires=[
    ],
    url='',
    long_description=open('README.md').read(),
    zip_safe=False,
)

