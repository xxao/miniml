#  Created by Martin Strohalm

from setuptools import setup, find_packages

# get version
from miniml import version
version = '.'.join(str(x) for x in version)

# get description
with open("README.md", "r") as fh:
    long_description = fh.read()

# set classifiers
classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3 :: Only',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Intended Audience :: Science/Research']

# main setup
setup(
    name = 'miniml',
    version = version,
    description = 'Provides very basic machine learning tools.',
    url = 'https://github.com/xxao/miniml',
    author = 'Martin Strohalm',
    author_email = '',
    license = 'MIT',
    packages = find_packages(),
    classifiers = classifiers,
    install_requires = ['numpy'],
    zip_safe = False)
