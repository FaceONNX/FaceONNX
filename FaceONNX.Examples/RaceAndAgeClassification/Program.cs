using FaceONNX;
using System;
using System.Drawing;
using System.IO;

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

            _faceDetectorLight = new FaceDetectorLight();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceRaceClassifier = new FaceRaceClassifier();
            _faceAgeClassifier = new FaceAgeClassifier();
            Directory.CreateDirectory(path);

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
                    var pen = new Pen(Color.Yellow, bitmap.Height / 200 + 1);
                    var font = new Font("Arial", 24);

                    Console.Write($"\t[Face #{i++}]: ");

                    var labels = GetRaceAndAge(bitmap, face);
                    Imaging.Draw(bitmap, pen, font, new Rectangle[] { face }, labels);
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
            using var cropped = Imaging.Crop(image, face);
            var points = _faceLandmarksExtractor.Forward(cropped);
            using var aligned = FaceLandmarksExtractor.Align(cropped, points);
            var race = _faceRaceClassifier.Forward(aligned);
            var raceLabel = FaceRaceClassifier.Labels[race.Argmax()];
            var age = _faceAgeClassifier.Forward(aligned);
            var ageLabel = FaceAgeClassifier.Labels[age.Argmax()];

            Console.WriteLine($"--> classified as [{raceLabel}] race and [{ageLabel}] age");

            return new string[] { raceLabel, ageLabel };
        }
    }
}
