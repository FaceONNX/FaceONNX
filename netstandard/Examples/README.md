<p align="center"><img width="25%" src="../FaceONNX/FaceONNX.png" /></p>
<p align="center"> Face analytics library based on deep neural networks and <b>ONNX</b> runtime </p>  

# Face detection
Build and run [FaceDetection.csproj](FaceDetection) to produce face detection results.
```
Image: [oscar.jpg] --> detected [53] faces
Image: [oscar2.jpg] --> detected [4] faces
Image: [selfie.jpg] --> detected [5] faces
Image: [selfie2.jpg] --> detected [22] faces
```

<p align="center"><img width="70%" src="FaceDetection/results/oscar2.jpg" /></p>
<p align="center"><b>Figure 1.</b> Results for <i>oscar2.jpg</i></p>  

# Face landmarks extraction
Build and run [FaceLandmarksExtraction.csproj](FaceLandmarksExtraction) to produce faces landmarks.
```
Image: [bruce.jpg] --> detected [1] faces
Image: [jake.jpg] --> detected [1] faces
Image: [kid.jpg] --> detected [1] faces
```
<p align="center"><img width="70%" src="FaceLandmarksExtraction/results/kid.jpg" /></p>
<p align="center"><b>Figure 2.</b> Results for <i>kid.jpg</i></p>  

# Face embeddings classification
Build and run [FaceEmbeddingsClassification.csproj](FaceEmbeddingsClassification) to classify faces as "Brad Pitt", "Nicole Kidman" or "Sarah Paulson".
```
Image: [brad_1.jpg] --> classified as [Brad Pitt] with similarity [0.91970295]
Image: [brad_2.jpg] --> classified as [Brad Pitt] with similarity [0.76870275]
Image: [brad_3.jpg] --> classified as [Brad Pitt] with similarity [0.77349997]
Image: [nicole_1.jpg] --> classified as [Nicole Kidman] with similarity [0.72006327]
Image: [nicole_2.jpg] --> classified as [Nicole Kidman] with similarity [0.66726977]
Image: [nicole_3.jpg] --> classified as [Nicole Kidman] with similarity [0.687553]
Image: [sarah_1.jpg] --> classified as [Sarah Paulson] with similarity [0.8746667]
Image: [sarah_2.jpg] --> classified as [Sarah Paulson] with similarity [0.8029]
Image: [sarah_3.jpg] --> classified as [Sarah Paulson] with similarity [0.8819007]
```

# Antispoofing depth classification
Build and run [AntispoofingDepthClassification.csproj](AntispoofingDepthClassification) to classify face depth as "Fake" or "Real".
```
Image: [fake_1.jpeg] --> classified as [Fake] with probability [0.9795725]
Image: [fake_10.jpeg] --> classified as [Fake] with probability [0.9999999]
Image: [fake_2.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_3.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_4.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_5.jpeg] --> classified as [Fake] with probability [0.9999993]
Image: [fake_6.jpeg] --> classified as [Fake] with probability [0.9999995]
Image: [fake_7.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_8.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_9.jpeg] --> classified as [Fake] with probability [0.99999833]
Image: [real_1.jpeg] --> classified as [Real] with probability [0.99999976]
Image: [real_10.jpeg] --> classified as [Real] with probability [0.99999976]
Image: [real_2.jpeg] --> classified as [Real] with probability [0.9999925]
Image: [real_3.jpeg] --> classified as [Real] with probability [0.9999993]
Image: [real_4.jpeg] --> classified as [Real] with probability [0.999997]
Image: [real_5.jpeg] --> classified as [Real] with probability [0.99999917]
Image: [real_6.jpeg] --> classified as [Real] with probability [1]
Image: [real_7.jpeg] --> classified as [Real] with probability [0.9998418]
Image: [real_8.jpeg] --> classified as [Real] with probability [0.9999691]
Image: [real_9.jpeg] --> classified as [Real] with probability [1]
```

# Eye blink detection
Build and run [EyeBlinkDetection.csproj](EyeBlinkDetection) to detect eye blink.
```
Image: [closed_closed.jpg] --> detected [1] faces
Image: [closed_open.jpg] --> detected [1] faces
Image: [open_open.jpg] --> detected [1] faces
```

<p align="center"><img width="70%" src="EyeBlinkDetection/results/open_open.jpg" /></p>
<p align="center"><b>Figure 3.</b> Results for <i>open_open.jpg</i></p> 

# Age & gender classification
Build and run [AgeGenderClassification.csproj](AgeGenderClassification) to classify faces as "Male" or "Female".
```
Image: [CF600.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.99982905] and [34.26622] ages
Image: [CF601.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.9999999] and [29.690014] ages
Image: [CF602.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.9056231] and [16.416363] ages
Image: [CF603.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.99966764] and [14.32959] ages
Image: [CF604.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.99980444] and [27.682442] ages
Image: [CM722.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [0.99912256] and [21.641447] ages
Image: [CM725.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [1] and [32.153343] ages
Image: [CM726.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [1] and [22.829079] ages
Image: [CM739.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [0.9999249] and [30.387539] ages
Image: [CM742.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [1] and [24.65458] ages
```

# Emotion & beauty estimation
Build and run [EmotionAndBeautyEstimation.csproj](EmotionAndBeautyEstimation) to classify face emotion and estimate face beauty. 
```
Image: [CF600.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [7.9/10.0] beauty
Image: [CF601.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [6.5/10.0] beauty
Image: [CF602.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [8.2/10.0] beauty
Image: [CF603.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [8.1/10.0] beauty
Image: [CF604.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [7.5/10.0] beauty
Image: [CM722.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [9.3/10.0] beauty
Image: [CM725.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [6.2/10.0] beauty
Image: [CM726.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [6.9/10.0] beauty
Image: [CM739.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [7.9/10.0] beauty
Image: [CM742.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [8.3/10.0] beauty
```

# GPU Perfomance tests
Build and run [GPUPerfomanceTests.csproj](GPUPerfomanceTests) to test FaceONNX inference on GPU.  
GPU Perfomance tests with CUDA 11.0.2 and cuDNN 8.0.4.30 (Windows 10) on NVIDIA GeForce GTX 1050 Ti (GPU) and Intel Core i7 9700K (CPU).
```
FaceONNX: GPU Perfomance tests with CUDA provider

Configuring FaceRecognitionTest
Configuring GPU device
Finished in [15641] ms
Running test for [100] iterations
Average time --> [15.78] ms
FPS --> [63.371357]
Finished in [1578] ms


Configuring FaceRecognitionTest
Configuring CPU device
Finished in [406] ms
Running test for [100] iterations
Average time --> [131.1] ms
FPS --> [7.6277647]
Finished in [13110] ms
```
