<p align="center"><img width="25%" src="docs/FaceONNX.png" /></p>
<p align="center"> Face analytics library based on deep neural networks and <b>ONNX</b> runtime </p>  

# FaceONNX
**FaceONNX** is a face analytics library based on [ONNX](https://onnx.ai/) runtime. It containts ready-made deep neural networks for face
* detection and landmarks extraction,
* gender and age classification,
* emotion and beauty classification,
* embeddings comparison and etc.  

# Version
You can build **FaceONNX** from sources or install to your own project using nuget package manager.
| Assembly | Specification | OS | Platform | Package | Algebra |
|-------------|:-------------:|:-------------:|:--------------:|:--------------:|:--------------:|
| [FaceONNX](FaceONNX) | .NET Standard 2.0 | Cross-platform | CPU | [NuGet](https://www.nuget.org/packages/FaceONNX/) | [UMapx](https://github.com/asiryan/UMapx) |
| [FaceONNX.Addons](FaceONNX.Addons) | .NET Standard 2.0 | Cross-platform | CPU | [NuGet](https://www.nuget.org/packages/FaceONNX.Addons/) | [UMapx](https://github.com/asiryan/UMapx) |
| [FaceONNX.Gpu](FaceONNX.Gpu) | .NET Standard 2.0 | Cross-platform | GPU | [NuGet](https://www.nuget.org/packages/FaceONNX.Gpu/) | [UMapx](https://github.com/asiryan/UMapx) |
| [FaceONNX.Addons.Gpu](FaceONNX.Addons.Gpu) | .NET Standard 2.0 | Cross-platform | GPU | [NuGet](https://www.nuget.org/packages/FaceONNX.Addons.Gpu/) | [UMapx](https://github.com/asiryan/UMapx) |

# Installation
C# interface  
```c#
using FaceONNX;
```
To get started with **FaceONNX**, it is recommended to look at the folder with [examples](Examples).  

# Models
**FaceONNX** is an open-source software. If you want to build **FaceONNX** from sources or use some features separately you can download already-made models from [GitHub](https://github.com/FaceONNX/FaceONNX.Models) repository or [Google.Drive](https://drive.google.com/drive/folders/1zfzHNeGju1r1-5vishZ--uaQNSorA0SJ?usp=sharing) disk.  

# References
[Python implementation](https://github.com/FaceONNX/pyfaceonnx)

# License
**FaceONNX** is released under the [MIT](LICENSE) license.
