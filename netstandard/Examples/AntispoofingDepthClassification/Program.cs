using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using UMapx.Core;

namespace AntispoofingDepthClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("FaceONNX: Antispoofing depth classification");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            using var faceDepthClassifier = new FaceDepthClassifier();
            var labels = FaceDepthClassifier.Labels;

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var output = faceDepthClassifier.Forward(bitmap);
                var max = Matrice.Max(output, out int gender);
                var label = labels[gender];

                Console.WriteLine($"Image: [{filename}] --> classified as [{label}] gender with probability [{output.Max()}]");
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
