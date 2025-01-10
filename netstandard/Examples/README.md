<p align="center"><img width="25%" src="../FaceONNX/FaceONNX.png" /></p>
<p align="center"> Face analytics library based on deep neural networks and <b>ONNX</b> runtime </p>  

# Face detection
Build and run [FaceDetection.csproj](FaceDetection) to produce face detection results.
```
Image: [oscar.jpg] --> detected [54] faces
Image: [oscar2.jpg] --> detected [4] faces
Image: [selfie.jpg] --> detected [5] faces
Image: [selfie2.jpg] --> detected [22] faces
Done.
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
Image: [brad_1.jpg] --> classified as [Brad Pitt] with similarity [0.9177492]
Image: [brad_2.jpg] --> classified as [Brad Pitt] with similarity [0.7718341]
Image: [brad_3.jpg] --> classified as [Brad Pitt] with similarity [0.78546345]
Image: [nicole_1.jpg] --> classified as [Nicole Kidman] with similarity [0.7083851]
Image: [nicole_2.jpg] --> classified as [Nicole Kidman] with similarity [0.66077006]
Image: [nicole_3.jpg] --> classified as [Nicole Kidman] with similarity [0.6967059]
Image: [sarah_1.jpg] --> classified as [Sarah Paulson] with similarity [0.86329275]
Image: [sarah_2.jpg] --> classified as [Sarah Paulson] with similarity [0.7996395]
Image: [sarah_3.jpg] --> classified as [Sarah Paulson] with similarity [0.8606752]
```

# Antispoofing depth classification
Build and run [AntispoofingDepthClassification.csproj](AntispoofingDepthClassification) to classify face depth as "Fake" or "Real".
```
Image: [fake_1.jpeg] --> classified as [Fake] with probability [0.9840763]
Image: [fake_10.jpeg] --> classified as [Fake] with probability [0.9999999]
Image: [fake_2.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_3.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_4.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_5.jpeg] --> classified as [Fake] with probability [0.9999995]
Image: [fake_6.jpeg] --> classified as [Fake] with probability [0.9999995]
Image: [fake_7.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_8.jpeg] --> classified as [Fake] with probability [1]
Image: [fake_9.jpeg] --> classified as [Fake] with probability [0.99999726]
Image: [real_1.jpeg] --> classified as [Real] with probability [0.99999976]
Image: [real_10.jpeg] --> classified as [Real] with probability [0.9999999]
Image: [real_2.jpeg] --> classified as [Real] with probability [0.99999285]
Image: [real_3.jpeg] --> classified as [Real] with probability [0.9999994]
Image: [real_4.jpeg] --> classified as [Real] with probability [0.9999975]
Image: [real_5.jpeg] --> classified as [Real] with probability [0.9999989]
Image: [real_6.jpeg] --> classified as [Real] with probability [1]
Image: [real_7.jpeg] --> classified as [Real] with probability [0.9997967]
Image: [real_8.jpeg] --> classified as [Real] with probability [0.9999814]
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
        [Face #1]: --> classified as [Female] gender with probability [0.9998264] and [34.42393] ages
Image: [CF601.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.99999976] and [30.10187] ages
Image: [CF602.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.9111187] and [15.594303] ages
Image: [CF603.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.9989963] and [15.32464] ages
Image: [CF604.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Female] gender with probability [0.9995233] and [26.72597] ages
Image: [CM722.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [0.9995109] and [21.628685] ages
Image: [CM725.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [0.9999999] and [33.206005] ages
Image: [CM726.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [1] and [22.117985] ages
Image: [CM739.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [0.9998927] and [30.164104] ages
Image: [CM742.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Male] gender with probability [1] and [24.632973] ages
```

# Emotion & beauty estimation
Build and run [EmotionAndBeautyEstimation.csproj](EmotionAndBeautyEstimation) to classify face emotion and estimate face beauty. 
```
Image: [CF600.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [7.9/10.0] beauty
Image: [CF601.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [6.7/10.0] beauty
Image: [CF602.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [8.2/10.0] beauty
Image: [CF603.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [8.1/10.0] beauty
Image: [CF604.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [7.4/10.0] beauty
Image: [CM722.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [9.6/10.0] beauty
Image: [CM725.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [6.3/10.0] beauty
Image: [CM726.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [6.8/10.0] beauty
Image: [CM739.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Happiness] emotion and [7.9/10.0] beauty
Image: [CM742.jpg] --> detected [1] faces
        [Face #1]: --> classified as [Neutral] emotion and [8.4/10.0] beauty
```

# GPU Perfomance tests
Build and run [GPUPerfomanceTests.csproj](GPUPerfomanceTests) to test FaceONNX inference on GPU.  
GPU Perfomance tests with CUDA 11.8.89 and cuDNN 9.6.0 (Windows 10) on NVIDIA GeForce RTX 4070 Ti (GPU) and Intel Core i7 13700KF (CPU).
```
FaceONNX: GPU Perfomance tests with CUDA provider

Configuring FaceRecognitionTest
Configuring GPU device
Finished in [359] ms
Running test for [100] iterations
Average time --> [6.56] ms
FPS --> [152.43903]
Finished in [656] ms


Configuring FaceRecognitionTest
Configuring CPU device
Finished in [250] ms
Running test for [100] iterations
Average time --> [67,19] ms
FPS --> [14,883166]
Finished in [6719] ms
```
