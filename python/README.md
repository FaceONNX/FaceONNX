<p align="center"><img width="25%" src="FaceONNX.png" /></p>
<p align="center"> Face analytics library based on deep neural networks and <b>ONNX</b> runtime </p>  

# FaceONNX
To build **FaceONNX** from sources for Python support follow this steps
```
git clone https://github.com/asiryan/FaceONNX
```
Download models from [Google.Drive](https://drive.google.com/drive/folders/1gh1E0yWqgzRX3Cxsp_EtZ2BAVOxyVAPb?usp=sharing) and place them to your [models folder](faceonnx/models).  
Install **FaceONNX** to your Python environment.  
```
python setup.py install
```
Python interface  
```python
import faceonnx
```
Python version of **FaceONNX** does not contains special tools for processing tensors, applying functions and painting recongition results. This is due to the fact that the python version uses opencv-python** and numpy.  
To get started with **FaceONNX**, it is recommended to look at the repository with [examples](examples).  

# License
**FaceONNX** is released under the [MIT](LICENSE) license.
