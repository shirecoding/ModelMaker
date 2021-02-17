from distutils.core import setup

exec(open("{{package_name}}/version.py").read())
setup(
    name="{{ package_name }}",
    version=__version__,
    author="",
    install_requires=["numpy", "opencv-python", "tensorflow-datasets"],
    url="",
    long_description=open("README.md").read(),
    zip_safe=False,
)
