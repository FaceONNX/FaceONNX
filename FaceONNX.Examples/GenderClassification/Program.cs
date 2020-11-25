using FaceONNX;
using FaceONNX.Core;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace GenderClassification
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Gender classification");
            var files = Directory.GetFiles(@"..\..\..\images");
            using var faceDetectorLight = new FaceDetectorLight();
            using var faceGenderClassifier = new FaceGenderClassifier();
            var labels = FaceGenderClassifier.Labels;

            foreach (var label in labels)
            {
                Directory.CreateDirectory(label);
            }

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = faceDetectorLight.Forward(bitmap);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    Console.Write($"\t[Face #{i++}]: ");

                    var output = faceGenderClassifier.Forward(bitmap);
                    var gender = Vector.Argmax(output);
                    var label = labels[gender];

                    Console.WriteLine($"--> classified as [{label}] gender with probability [{output.Max()}]");
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
