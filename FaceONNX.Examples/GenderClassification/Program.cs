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
                var output = faceGenderClassifier.Forward(bitmap);
                var gender = Vector.Argmax(output);
                var filename = Path.GetFileName(file);
                var label = labels[gender];

                Console.WriteLine($"Image: [{filename}] --> classified as [{label}] with probability [{output.Max()}]");
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
