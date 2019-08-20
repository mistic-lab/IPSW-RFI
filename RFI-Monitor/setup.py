from setuptools import setup, find_packages
import sys

setup(name='RFI_Localisation',
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'tensorflow',
    'scikit-image',
    'sklearn'],
    description='Tool used to localise RFI signals',
    author='Rory Coles')
