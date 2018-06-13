#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs pyGAM package

Can be run via command: python setup.py install (or python setup.py develop)
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()
    requirements = [req for req in requirements.split('\n') if len(req) > 0]

args = dict(
    name='pyGAM',
    long_description=readme,
    license=license,
    version='0.5.2',
    packages=find_packages(exclude=('datasets', 'imgs', 'tests')),
    install_requires=requirements
)

setup(**args)