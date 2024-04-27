using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using UMapx.Core;

namespace AgeGenderClassification
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Age and gender classification");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            using var faceDetector = new FaceDetector();
            using var faceLandmarksExtractor = new Face68LandmarksExtractor();
            using var faceGenderClassifier = new FaceGenderClassifier();
            using var faceAgeEstimator = new FaceAgeEstimator();
            var labels = FaceGenderClassifier.Labels;

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

                    var output = faceGenderClassifier.Forward(aligned);
                    var max = Matrice.Max(output, out int gender);
                    var label = labels[gender];
                    var age = faceAgeEstimator.Forward(aligned);

                    Console.WriteLine($"--> classified as [{label}] gender with probability [{output.Max()}] and [{age.First()}] ages");
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
