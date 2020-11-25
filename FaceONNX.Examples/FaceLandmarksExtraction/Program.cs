using FaceONNX;
using FaceONNX.Core;
using System;
using System.Drawing;
using System.IO;

namespace FaceLandmarksExtraction
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face landmarks extraction");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            using var faceDetector = new FaceDetector();
            using var faceLandmarksExtractor = new FaceLandmarksExtractor();
            var painter = new Painter()
            {
                PointPen = new Pen(Color.Yellow, 4),
                Transparency = 0,
            };
            
            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = faceDetector.Forward(bitmap);
                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    var points = faceLandmarksExtractor.Forward(bitmap, face);
                    
                    foreach (var point in points)
                    {
                        var paintData = new PaintData()
                        {
                            Points = point,
                            Title = string.Empty,
                        };

                        painter.Draw(bitmap, paintData);
                        bitmap.Save(Path.Combine(path, filename));
                    }
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
