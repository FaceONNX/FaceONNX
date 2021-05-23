using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using UMapx.Core;
using UMapx.Imaging;
using UMapx.Visualization;

namespace RaceAndAgeClassification
{
    class Program
    {
        static FaceDetectorLight _faceDetectorLight;
        static FaceLandmarksExtractor _faceLandmarksExtractor;
        static FaceRaceClassifier _faceRaceClassifier;
        static FaceAgeClassifier _faceAgeClassifier;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Race and age classification");
            var files = Directory.GetFiles(@"..\..\..\images");
            var path = @"..\..\..\results";
            Directory.CreateDirectory(path);

            _faceDetectorLight = new FaceDetectorLight();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceRaceClassifier = new FaceRaceClassifier();
            _faceAgeClassifier = new FaceAgeClassifier();
            var painter = new Painter()
            {
                PointPen = new Pen(Color.Yellow, 4),
                Transparency = 0,
                TextFont = new Font("Arial", 24)
            };

            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var filename = Path.GetFileName(file);
                var faces = _faceDetectorLight.Forward(bitmap);
                int i = 1;

                Console.WriteLine($"Image: [{filename}] --> detected [{faces.Length}] faces");

                foreach (var face in faces)
                {
                    Console.Write($"\t[Face #{i++}]: ");

                    var labels = GetRaceAndAge(bitmap, face);

                    var paintData = new PaintData()
                    {
                        Rectangle = face,
                        Labels = labels
                    };

                    using var graphics = Graphics.FromImage(bitmap);
                    painter.Draw(graphics, paintData);
                }

                bitmap.Save(Path.Combine(path, filename));
            }

            _faceDetectorLight.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceRaceClassifier.Dispose();
            _faceAgeClassifier.Dispose();

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        static string[] GetRaceAndAge(Bitmap image, Rectangle face)
        {
            using var cropped = BitmapTransform.Crop(image, face);
            var points = _faceLandmarksExtractor.Forward(cropped);
            using var aligned = FaceLandmarksExtractor.Align(cropped, points);
            var race = _faceRaceClassifier.Forward(aligned);
            var maxRace = Matrice.Max(race, out int argmaxRace);
            var raceLabel = FaceRaceClassifier.Labels[argmaxRace];
            var age = _faceAgeClassifier.Forward(aligned);
            var maxAge = Matrice.Max(age, out int argmaxAge);
            var ageLabel = FaceAgeClassifier.Labels[argmaxAge];

            Console.WriteLine($"--> classified as [{raceLabel}] race and [{ageLabel}] age");

            return new string[] { raceLabel, ageLabel };
        }
    }
}
