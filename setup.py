from setuptools import setup, find_packages

from autoimplant import __version__

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'autoimplant'

setup(
    name=name,
    version=__version__,
    packages=find_packages(include=(name,)),
    descriprion='Repository of the course project during Deep Learning class at Skoltech (spring 2021)',
    install_requires=requirements
)
