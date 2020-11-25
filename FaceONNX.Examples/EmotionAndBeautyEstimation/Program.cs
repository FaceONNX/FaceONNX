using FaceONNX;
using FaceONNX.Core;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace EmotionAndBeautyEstimation
{
    class Program
    {
        static FaceDetectorLight _faceDetectorLight;
        static FaceLandmarksExtractor _faceLandmarksExtractor;
        static FaceEmotionClassifier _faceEmotionClassifier;
        static FaceBautyClassifier _faceBautyClassifier;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Emotion and beauty estimation");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            _faceDetectorLight = new FaceDetectorLight();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceEmotionClassifier = new FaceEmotionClassifier();
            _faceBautyClassifier = new FaceBautyClassifier();
            var painter = new Painter()
            {
                BoxPen = new Pen(Color.Red, 4),
                Transparency = 0,
                TextFont = new Font("Arial", 12)
            };

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var size = bitmap.Size;
                var offset = 100;
                using var template = Imaging.CreateBitmap(
                    new Size(size.Width + offset, size.Height + offset), 
                    Color.White);

                template.Merge(bitmap, new Rectangle(offset / 2, offset / 2, bitmap.Size.Width, bitmap.Size.Height));
                var filename = Path.GetFileName(file);
                var faces = _faceDetectorLight.Forward(template);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    Console.Write($"\t[Face #{i++}]: ");

                    var labels = GetEmotionAndBeauty(template, face);

                    var paintData = new PaintData()
                    {
                        Labels = labels,
                        Rectangle = face
                    };

                    painter.Draw(template, paintData);
                }

                template.Save(Path.Combine(path, filename));
            }

            _faceDetectorLight.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceEmotionClassifier.Dispose();
            _faceBautyClassifier.Dispose();

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        static string[] GetEmotionAndBeauty(Bitmap image, Rectangle face)
        {
            using var cropped = Imaging.Crop(image, face);
            var points = _faceLandmarksExtractor.Forward(cropped);
            using var aligned = FaceLandmarksExtractor.Align(cropped, points);
            var emotion = _faceEmotionClassifier.Forward(aligned);
            var emotionLabel = FaceEmotionClassifier.Labels[emotion.Argmax()];
            var beauty = _faceBautyClassifier.Forward(aligned);
            var beautyLabel = $"{Math.Round(2 * beauty.Max(), 1)}/10.0";

            Console.WriteLine($"--> classified as [{emotionLabel}] emotion and [{beautyLabel}] beauty");

            return new string[] { emotionLabel, beautyLabel };
        }
    }
}
