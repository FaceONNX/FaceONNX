# Face detection
Build and start [**FaceDetection.csproj**](FaceDetection) project to produce face detection results
```batch
FaceONNX: Face detection
Processing 4 images
Image: [oscar.jpg] --> detected [13] faces
Image: [oscar2.jpg] --> detected [4] faces
Image: [selfie.jpg] --> detected [5] faces
Image: [selfie2.jpg] --> detected [18] faces
Done.
```

<p align="center"><img width="70%" src="FaceDetection/results/oscar2.jpg" /></p>
<p align="center"><b>Figure 1.</b> Results for oscar2.jpg</p>  

# Face embeddings classification
Build and start [**FaceEmbeddingsClassification.csproj**](FaceEmbeddingsClassification) project to classify 6 faces to Brad Pitt, Nicole Kidman and Sarah Paulson
```batch
FaceONNX: Face embeddings classification
Embeddings count: 3
Processing 9 images
Image: [brad_1.jpg] --> classified as [Brad Pitt] with similarity [0,96517545]
Image: [brad_2.jpg] --> classified as [Brad Pitt] with similarity [0,8623285]
Image: [brad_3.jpg] --> classified as [Brad Pitt] with similarity [0,8151555]
Image: [nicole_1.jpg] --> classified as [Nicole Kidman] with similarity [0,8584538]
Image: [nicole_2.jpg] --> classified as [Nicole Kidman] with similarity [0,8298364]
Image: [nicole_3.jpg] --> classified as [Nicole Kidman] with similarity [0,8977611]
Image: [sarah_1.jpg] --> classified as [Sarah Paulson] with similarity [0,9042954]
Image: [sarah_2.jpg] --> classified as [Sarah Paulson] with similarity [0,8464778]
Image: [sarah_3.jpg] --> classified as [Sarah Paulson] with similarity [0,84192747]
Done.
```
