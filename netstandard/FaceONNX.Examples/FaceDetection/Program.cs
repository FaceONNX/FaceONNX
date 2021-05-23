using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using UMapx.Visualization;

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

            using var faceDetectorLight = new FaceDetectorLight(0.9f, 0.5f);
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
                    using var graphics = Graphics.FromImage(bitmap);
                    painter.Draw(graphics, paintData);
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
