import setuptools
import os

# package metadata
NAME = 'faceonnx'
VERSION = '1.0.5.2'
DESCRIPTION = 'Face analytics library based on deep neural networks and ONNX runtime.'
LICENSE = 'MIT'
GIT = 'https://github.com/asiryan/FaceONNX'
PYTHON = '>=3.5'

# directory
this = os.path.dirname(__file__)

# readme
with open(os.path.join(this, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# setup tools
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license=LICENSE,
    packages=setuptools.find_packages(),
    python_requires=PYTHON,
    author='Valery Asiryan',
    author_email='dmc5mod@yandex.ru',
    url=GIT,
    install_requires=[
        'numpy',
        'opencv-python',
        'onnx >= 1.4.0',
        'onnxruntime >= 1.6.0',
        'googledrivedownloader'
    ],
    classifiers=[
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)