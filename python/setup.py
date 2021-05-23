import setuptools
import os

# package metadata
NAME = 'faceonnx'
VERSION = '1.0.5.1'
DESCRIPTION = 'Face analytics library based on deep neural networks and ONNX runtime.'
LICENSE = 'MIT'
GIT = 'https://github.com/asiryan/FaceONNX'
PYTHON = '>=3.5'

# directory
this = os.path.dirname(__file__)

# download models
import subprocess
import sys

try:
    import google_drive_downloader
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'googledrivedownloader'])

from google_drive_downloader import GoogleDriveDownloader as g

def download(path):
    """
    Downloads ONNX models from google.drive.
    Args:
        path: Path
    """
    g.download_file_from_google_drive(file_id='1T39Qh7FA-tNbDge6PE4gN-8Ajgnmozum', dest_path=f'.{path}/faceonnx/models/age_googlenet.onnx')
    g.download_file_from_google_drive(file_id='1Eqr3KXXEFI2vhFAggknmNdmKtO0Ap_0C', dest_path=f'.{path}/faceonnx/models/beauty_resnet18.onnx')
    g.download_file_from_google_drive(file_id='1Oqd-0klyn-loAnUyXdah4FN131YfDFcv', dest_path=f'.{path}/faceonnx/models/emotion_cnn.onnx')
    g.download_file_from_google_drive(file_id='1U6uKXWCPmxiShhCnXfBQZkl97Gm3GEJV', dest_path=f'.{path}/faceonnx/models/face_detector_320.onnx')
    g.download_file_from_google_drive(file_id='1tB7Y5l5Jf2270IisgSZ3rbCQs0-pkNIQ', dest_path=f'.{path}/faceonnx/models/face_detector_640.onnx')
    g.download_file_from_google_drive(file_id='1ZyzRGsQLpEVkzIVQn1qXwdPv0gxfj0Ch', dest_path=f'.{path}/faceonnx/models/face_unet_256.onnx')
    g.download_file_from_google_drive(file_id='1ouxERxbMSZpH6FA-y7mrgoVMuTTtig3P', dest_path=f'.{path}/faceonnx/models/face_unet_512.onnx')
    g.download_file_from_google_drive(file_id='1ouxERxbMSZpH6FA-y7mrgoVMuTTtig3P', dest_path=f'.{path}/faceonnx/models/face_unet_512.onnx')
    g.download_file_from_google_drive(file_id='1ZsqnXunyEgxaAx9WoX5uQv_T7RWvTbTz', dest_path=f'.{path}/faceonnx/models/gender_googlenet.onnx')
    g.download_file_from_google_drive(file_id='1qgM6ZqMyB60FYlzzxNDyUefifLS0lhag', dest_path=f'.{path}/faceonnx/models/landmarks_68_pfld.onnx')
    g.download_file_from_google_drive(file_id='1b5KC_qG-mTSCkM2vW4VsThETzRkmc7Fa', dest_path=f'.{path}/faceonnx/models/race_googlenet.onnx')
    g.download_file_from_google_drive(file_id='1ijbMt1LETLQc6GDGAtEJx8ggEGenyM7m', dest_path=f'.{path}/faceonnx/models/recognition_resnet27.onnx')
    g.download_file_from_google_drive(file_id='1PHGb4d0ews2jNBGBaYZ4HWNJzph00oJa', dest_path=f'.{path}/faceonnx/models/mask_googlenet_slim.onnx')

download(this)

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
        'onnxruntime >= 1.6.0'
    ],
    package_data={
      NAME: ['*.onnx', os.path.join(this, './models/*.onnx')],
    },
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