using FaceONNX;
using FaceONNX.Core;
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
            Directory.CreateDirectory(path);

            using var faceDetectorLight = new FaceDetectorLight(0.95f, 0.25f);
            var painter = new Painter()
            {
                BoxPen = new Pen(Color.Yellow, 4),
                Transparency = 0,
            };

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var output = faceDetectorLight.Forward(bitmap);

                foreach (var rectangle in output)
                {
                    var paintData = new PaintData()
                    {
                        Rectangle = rectangle,
                        Title = string.Empty
                    };
                    painter.Draw(bitmap, paintData);
                }

                var filename = Path.GetFileName(file);
                bitmap.Save(Path.Combine(path, filename));
                Console.WriteLine($"Image: [{filename}] --> detected [{output.Length}] faces");
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
