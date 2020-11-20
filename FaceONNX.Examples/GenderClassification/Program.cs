using System;
using System.Drawing;
using System.IO;
using FaceONNX;

namespace GenderClassification
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Gender classification");
            var files = Directory.GetFiles(@"..\..\..\images");
            var faceGenderClassifier = new FaceGenderClassifier();
            var labels = FaceGenderClassifier.Labels;

            foreach (var label in labels)
            {
                Directory.CreateDirectory(label);
            }

            Console.WriteLine($"Classifying {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var output = faceGenderClassifier.Forward(bitmap);
                var gender = Vector.Argmax(output);
                var filename = Path.GetFileName(file);
                var label = labels[gender];

                File.Copy(file, Path.Combine(label, filename), true);
                Console.WriteLine($"Image: {filename} --> classified as {label}");
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
