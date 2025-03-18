'''Setup/build/install script for osiris utils.'''

import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='osiris_utils',
    version='v1.1.1',
    description=('Utilities to manipulate and visualize OSIRIS framework output data'),
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author=['João Pedro Ferreira Biu', 'João Cândido'],
    author_email=['joaopedrofbiu@tecnico.ulisboa.pt'],
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
    ],
    keywords=['OSIRIS', 'particle-in-cell', 'plasma', 'physics'],
    packages=find_packages(exclude=['docs', 'tests', 'local', 'report']),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.11',
    project_urls={
        'Issues Tracker': 'https://github.com/joaopedrobiu6/osiris_utils/issues',
        'Source Code': 'https://github.com/joaopedrobiu6/osiris_utils',
    },
)