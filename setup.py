import os
import sysconfig
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

from autoimplant import __version__

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

name = 'autoimplant'


class WithExternal(build_ext):
    def run(self):
        os.chdir(os.getenv("HOME"))

        os.system(f'git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git')

        link_path = os.path.join(sysconfig.get_paths()["purelib"], 'resnets_3d')
        os.mkdir(link_path)

        os.system(f'ln -s ~/3D-ResNets-PyTorch/models {link_path}')

        build_ext.run(self)


setup(
    name=name,
    version=__version__,
    packages=find_packages(include=(name,)),
    cmdclass={'build_ext': WithExternal},
    descriprion='Repository of the course project during Deep Learning class at Skoltech (spring 2021)',
    install_requires=requirements
)
