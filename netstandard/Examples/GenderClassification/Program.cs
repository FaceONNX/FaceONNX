using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace GenderClassification
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Gender classification");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            using var faceDetector = new FaceDetector();
            using var faceGenderClassifier = new FaceGenderClassifier();
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

                    var cropped = BitmapTransform.Crop(bitmap, face);
                    var output = faceGenderClassifier.Forward(cropped);
                    var max = Matrice.Max(output, out int gender);
                    var label = labels[gender];

                    Console.WriteLine($"--> classified as [{label}] gender with probability [{output.Max()}]");
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
