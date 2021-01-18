<p align="center"><img width="25%" src="docs/FaceONNX.png" /></p>
<p align="center"> Face analytics library based on deep neural networks and <b>ONNX</b> runtime </p>  

# FaceONNX
**FaceONNX** is a face analytics library based on [ONNX](https://onnx.ai/) runtime. It containts ready-made deep neural networks for face
* detection and landmarks extraction,
* semantic segmentation,
* gender and race classification,
* age and emotion classification,
* beauty estimation,
* embeddings comparison and etc.  
  
**FaceONNX** basically oriented for [.NET platform](netstandard) (C#), but it has [Python](python) realization.  

# How to Use
You can build **FaceONNX** from sources or install to your own project using nuget package manager.
| Assembly | Type | Version | Package |
|:-------------|:-------------|:--------------|:--------------|
| FaceONNX | CPU | [1.0.2.1](FaceONNX/FaceONNX.csproj) | [Nuget](https://www.nuget.org/packages/FaceONNX/) |
| FaceONNX.Gpu | GPU | [1.0.2.1](FaceONNX/FaceONNX.Gpu.csproj)| [Nuget](https://www.nuget.org/packages/FaceONNX.Gpu/) |

To get started with **FaceONNX**, it is recommended to look at the repository with [examples](FaceONNX.Examples).  

# License
**FaceONNX** is released under the [MIT](LICENSE) license.
