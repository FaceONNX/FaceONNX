using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace EmotionAndBeautyEstimation
{
    class Program
    {
        static FaceDetector _faceDetector;
        static FaceLandmarksExtractor _faceLandmarksExtractor;
        static FaceEmotionClassifier _faceEmotionClassifier;
        static FaceBeautyClassifier _faceBeautyClassifier;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Emotion and beauty estimation");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            _faceDetector = new FaceDetector();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceEmotionClassifier = new FaceEmotionClassifier();
            _faceBeautyClassifier = new FaceBeautyClassifier();

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = _faceDetector.Forward(bitmap);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    Console.Write($"\t[Face #{i++}]: ");

                    var labels = GetEmotionAndBeauty(bitmap, face);
                }
            }

            _faceDetector.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceEmotionClassifier.Dispose();
            _faceBeautyClassifier.Dispose();

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        static string[] GetEmotionAndBeauty(Bitmap image, Rectangle face)
        {
            using var cropped = BitmapTransform.Crop(image, face);
            var points = _faceLandmarksExtractor.Forward(cropped);
            using var aligned = FaceLandmarksExtractor.Align(cropped, points);
            var emotion = _faceEmotionClassifier.Forward(aligned);
            var max = Matrice.Max(emotion, out int argmax);
            var emotionLabel = FaceEmotionClassifier.Labels[argmax];
            var beauty = _faceBeautyClassifier.Forward(aligned);
            var beautyLabel = $"{Math.Round(2 * beauty.Max(), 1)}/10.0";

            Console.WriteLine($"--> classified as [{emotionLabel}] emotion and [{beautyLabel}] beauty");

            return new string[] { emotionLabel, beautyLabel };
        }
    }
}
