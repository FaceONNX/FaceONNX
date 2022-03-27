using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using UMapx.Imaging;
using UMapx.Visualization;

namespace EyeBlinkDetection
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Eye blink detection");
            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            using var faceDetector = new FaceDetector();
            using var faceLandmarksExtractor = new FaceLandmarksExtractor();
            using var eyeBlinkClassifier = new EyeBlinkClassifier();

            using var painter = new Painter()
            {
                PointPen = new Pen(Color.Red, 4),
                Transparency = 0,
                TextFont = new Font("Arial", 4)
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
                    // crop and align face
                    using var cropped = BitmapTransform.Crop(bitmap, face);
                    var points = faceLandmarksExtractor.Forward(cropped);

                    // eye blink detection
                    var eyes = EyeBlinkClassifier.GetEyesRectangles(points);

                    var left_eye_rect = eyes.Item1;
                    var right_eye_rect = eyes.Item2;

                    using var left_eye = BitmapTransform.Crop(cropped, left_eye_rect);
                    using var right_eye = BitmapTransform.Crop(cropped, right_eye_rect);

                    var left_eye_value = eyeBlinkClassifier.Forward(left_eye);
                    var right_eye_value = eyeBlinkClassifier.Forward(right_eye);

                    // drawing face detection and
                    // landmarks extraction results
                    using var graphics = Graphics.FromImage(bitmap);

                    var paintData = new PaintData
                    {
                        Points = points.Add(face.GetPoint()),
                        Title = string.Empty,
                    };

                    painter.Draw(graphics, paintData);

                    // drawing eye bling detection results
                    var paintLeftEyeData = new PaintData
                    {
                        Rectangle = left_eye_rect.Add(face.GetPoint()),
                        Labels = ToString(left_eye_value)
                    };

                    var paintRightEyeData = new PaintData
                    {
                        Rectangle = right_eye_rect.Add(face.GetPoint()),
                        Labels = ToString(right_eye_value)
                    };

                    painter.Draw(graphics, paintLeftEyeData);
                    painter.Draw(graphics, paintRightEyeData);

                    bitmap.Save(Path.Combine(path, filename));
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        private static string[] ToString(float[] tensor)
        {
            var value = Math.Round(tensor[0], 1);
            return new string[] { value.ToString() };
        }
    }
}
