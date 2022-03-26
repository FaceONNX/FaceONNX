using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using UMapx.Imaging;
using UMapx.Visualization;

namespace FaceLandmarksExtraction
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face landmarks extraction");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            using var faceDetector = new FaceDetector();
            using var faceLandmarksExtractor = new FaceLandmarksExtractor();
            using var painter = new Painter()
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
                    // crop face
                    using var cropped = BitmapTransform.Crop(bitmap, face);
                    var points = faceLandmarksExtractor.Forward(cropped);

                    var paintData = new PaintData()
                    {
                        Points = points.Add(face.GetPoint()),
                        Title = string.Empty,
                    };

                    using var graphics = Graphics.FromImage(bitmap);
                    painter.Draw(graphics, paintData);
                    bitmap.Save(Path.Combine(path, filename));
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
