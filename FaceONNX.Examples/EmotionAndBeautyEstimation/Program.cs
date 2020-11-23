using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace EmotionAndBeautyEstimation
{
    class Program
    {
        static FaceDetectorLight _faceDetectorLight;
        static FaceLandmarksExtractor _faceLandmarksExtractor;
        static FaceEmotionClassifier _faceEmotionClassifier;
        static FaceBautyClassifier _faceBautyClassifier;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Emotion and beauty estimation");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";

            _faceDetectorLight = new FaceDetectorLight();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceEmotionClassifier = new FaceEmotionClassifier();
            _faceBautyClassifier = new FaceBautyClassifier();
            Directory.CreateDirectory(path);

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = _faceDetectorLight.Forward(bitmap);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    var pen = new Pen(Color.Yellow, bitmap.Height / 200 + 1);
                    var font = new Font("Arial", 24);

                    Console.Write($"\t[Face #{i++}]: ");

                    var labels = GetEmotionAndBeauty(bitmap, face);
                    Imaging.Draw(bitmap, pen, font, new Rectangle[] { face }, labels);
                }

                bitmap.Save(Path.Combine(path, filename));
            }

            _faceDetectorLight.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceEmotionClassifier.Dispose();
            _faceBautyClassifier.Dispose();

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        static string[] GetEmotionAndBeauty(Bitmap image, Rectangle face)
        {
            using var cropped = Imaging.Crop(image, face);
            var points = _faceLandmarksExtractor.Forward(cropped);
            using var aligned = FaceLandmarksExtractor.Align(cropped, points);
            var emotion = _faceEmotionClassifier.Forward(aligned);
            var emotionLabel = FaceEmotionClassifier.Labels[emotion.Argmax()];
            var beauty = _faceBautyClassifier.Forward(aligned);
            var beautyLabel = (Math.Round(2 * beauty.Max(), 1)).ToString();
            
            Console.WriteLine($"--> classified as [{emotionLabel}] emotion and [{beautyLabel}/10.0] beauty");

            return new string[] { emotionLabel, beautyLabel };
        }
    }
}
