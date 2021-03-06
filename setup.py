from distutils.core import setup

from setuptools import find_packages

exec(open("modelmaker/version.py").read())
setup(
    name="model-maker",
    version=__version__,
    author="shirecoding",
    author_email="shirecoding@gmail.com",
    scripts=["bin/modelmaker"],
    install_requires=["Jinja2"],
    url="https://github.com/shirecoding/ModelMaker",
    download_url=f"https://github.com/shirecoding/ModelMaker/archive/{__version__}.tar.gz",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    packages=find_packages() + ["modelmaker.resources"],
    package_data={
        "modelmaker.resources": [
            "templates/*",
            "templates/**/*",
            "templates/**/**/*",
            "templates/**/**/**/*",
            "templates/**/**/**/**/*",
        ]
    },
)
