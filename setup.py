# -*- coding:utf-8 -*-

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

home_url = 'https://github.com/DataCanvasIO/CausalLab'


def read_requirements(file_path='requirements.txt'):
    import os

    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r')as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

    return lines


def read_extra_requirements():
    import glob
    import re

    extra = {}

    for file_name in glob.glob('requirements-*.txt'):
        key = re.search('requirements-(.+).txt', file_name).group(1)
        req = read_requirements(file_name)
        if req:
            extra[key] = req

    if extra and 'all' not in extra.keys():
        extra['all'] = sorted({v for req in extra.values() for v in req})

    return extra



def read_description(file_path='README.md'):
    with open(file_path, encoding='utf-8') as f:
        desc = f.read()
    return desc


import causallab

version = causallab.__version__

MIN_PYTHON_VERSION = '>=3.8'

# long_description = open('README.md', encoding='utf-8').read()
long_description = read_description()

requires = read_requirements()
extras_require = read_extra_requirements()

setup(
    name='causallab',
    version=version,
    description='An Interactive Causal Analysis Tool',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=home_url,
    author='DataCanvas Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requires,
    python_requires=MIN_PYTHON_VERSION,
    extras_require=extras_require,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests*')),
    package_data={
    },
    entry_points={
    },
    zip_safe=False,
    include_package_data=True,
)
