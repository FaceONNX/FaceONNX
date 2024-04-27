using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using UMapx.Core;

namespace EmotionAndBeautyEstimation
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Emotion and beauty estimation");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            using var faceDetector = new FaceDetector();
            using var faceLandmarksExtractor = new Face68LandmarksExtractor();
            using var faceEmotionClassifier = new FaceEmotionClassifier();
            using var faceBeautyClassifier = new FaceBeautyClassifier();

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = faceDetector.Forward(bitmap);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    Console.Write($"\t[Face #{i++}]: ");

                    var box = face.Box;
                    var points = faceLandmarksExtractor.Forward(bitmap, box);
                    var angle = points.RotationAngle;
                    using var aligned = FaceProcessingExtensions.Align(bitmap, box, angle, false);
                    var emotion = faceEmotionClassifier.Forward(aligned);
                    var max = Matrice.Max(emotion, out int argmax);
                    var emotionLabel = FaceEmotionClassifier.Labels[argmax];
                    var beauty = faceBeautyClassifier.Forward(aligned);
                    var beautyLabel = $"{Math.Round(2 * beauty.Max(), 1)}/10.0";

                    Console.WriteLine($"--> classified as [{emotionLabel}] emotion and [{beautyLabel}] beauty");
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
