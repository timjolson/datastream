from setuptools import setup, find_packages

setup(
    name='datastream',
    version="0.7",
    packages=find_packages(),
    install_requires=['numpy'],
    tests_require=['pytest'],
)
