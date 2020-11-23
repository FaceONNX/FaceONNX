# Face detection
Build and start [**FaceDetection.csproj**](FaceDetection) to produce face detection results
```batch
Image: [oscar.jpg] --> detected [13] faces
Image: [oscar2.jpg] --> detected [4] faces
Image: [selfie.jpg] --> detected [5] faces
Image: [selfie2.jpg] --> detected [18] faces
```

<p align="center"><img width="70%" src="FaceDetection/results/oscar2.jpg" /></p>
<p align="center"><b>Figure 1.</b> Results for oscar2.jpg</p>  

# Face embeddings classification
Build and start [**FaceEmbeddingsClassification.csproj**](FaceEmbeddingsClassification) to classify 9 faces as Brad Pitt, Nicole Kidman or Sarah Paulson
```batch
Image: [brad_1.jpg] --> classified as [Brad Pitt] with similarity [0,96517545]
Image: [brad_2.jpg] --> classified as [Brad Pitt] with similarity [0,8623285]
Image: [brad_3.jpg] --> classified as [Brad Pitt] with similarity [0,8151555]
Image: [nicole_1.jpg] --> classified as [Nicole Kidman] with similarity [0,8584538]
Image: [nicole_2.jpg] --> classified as [Nicole Kidman] with similarity [0,8298364]
Image: [nicole_3.jpg] --> classified as [Nicole Kidman] with similarity [0,8977611]
Image: [sarah_1.jpg] --> classified as [Sarah Paulson] with similarity [0,9042954]
Image: [sarah_2.jpg] --> classified as [Sarah Paulson] with similarity [0,8464778]
Image: [sarah_3.jpg] --> classified as [Sarah Paulson] with similarity [0,84192747]
```

# Face landmarks extraction
Build and start [**FaceLandmarksExtraction.csproj**](FaceLandmarksExtraction) to produce face landmarks for 3 images
```batch
Image: [bruce.jpg] --> detected [1] faces
Image: [jake.jpg] --> detected [1] faces
Image: [kid.jpg] --> detected [1] faces
```
<p align="center"><img width="70%" src="FaceLandmarksExtraction/results/kid.jpg" /></p>
<p align="center"><b>Figure 2.</b> Results for kid.jpg</p>  

# Face semantic segmentation
Build and start [**FaceSemanticSegmentation.csproj**](FaceSemanticSegmentation) to produce face segmentation maps for 3 images
```batch
Image: [brad.jpg] --> detected [1] faces
Image: [michael.jpg] --> detected [1] faces
Image: [nicole.jpg] --> detected [1] faces
```
<p align="center"><img width="35%" src="FaceSemanticSegmentation/images/brad.jpg"/><img width="35%" src="FaceSemanticSegmentation/results/brad.jpg" /></p>
<p align="center"><b>Figure 3.</b> Original image brad.jpg and face segmentation map</p>  
