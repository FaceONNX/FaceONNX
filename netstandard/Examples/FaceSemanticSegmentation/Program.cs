using FaceONNX;
using System;
using System.Drawing;
using System.IO;

namespace FaceSemanticSegmentation
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face semantic segmentation");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";

            using var faceDetector = new FaceDetector();
            using var faceParser = new FaceParser();
            Directory.CreateDirectory(path);

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = faceDetector.Forward(bitmap);
                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    var labels = faceParser.Forward(bitmap, face);

                    foreach (var label in labels)
                    {
                        using var segmentated = FaceParser.ToBitmap(label);
                        segmentated.Save(Path.Combine(path, filename));
                    }
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
