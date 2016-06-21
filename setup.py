# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Radiation Oncology Case Creation',
    version='0.0.1',
    description='Creation of a ghost 2D cancer patient case',
    long_description=readme,
    author='Wilmer E. Henao',
    author_email='wilmer@umich.edu',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

