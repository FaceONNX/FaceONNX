using FaceONNX;
using System;
using System.Drawing;
using System.IO;

namespace FaceDetection
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face detection");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";

            using var faceDetectorLight = new FaceDetectorLight(0.75f, 0.25f);
            Directory.CreateDirectory(path);

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var output = faceDetectorLight.Forward(bitmap);
                var pen = new Pen(Color.Yellow, bitmap.Height / 200 + 1);
                Imaging.Draw(bitmap, pen, output);

                var filename = Path.GetFileName(file);
                bitmap.Save(Path.Combine(path, filename));
                Console.WriteLine($"Image: [{filename}] --> detected [{output.Length}] faces");
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
