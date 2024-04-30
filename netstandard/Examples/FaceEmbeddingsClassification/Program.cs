using FaceONNX;
using System;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;

namespace FaceEmbeddingsClassification
{
    class Program
    {
        static FaceDetector faceDetector;
        static Face68LandmarksExtractor _faceLandmarksExtractor;
        static FaceEmbedder _faceEmbedder;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Face embeddings classification");

            var imagePath = Environment.OSVersion.Platform == PlatformID.Unix ? 
                Path.Combine(".", "images", "fit") : @"..\..\..\images\fit";
            var fits = Directory.GetFiles(imagePath);

            faceDetector = new FaceDetector();
            _faceLandmarksExtractor = new Face68LandmarksExtractor();
            _faceEmbedder = new FaceEmbedder();
            var embeddings = new Embeddings();

            foreach (var fit in fits)
            {
                using var theImage = Image.Load<Rgb24>(fit);
                var embedding = GetEmbedding(theImage);
                var name = Path.GetFileNameWithoutExtension(fit);
                embeddings.Add(embedding, name);
            }

            Console.WriteLine($"Embeddings count: {embeddings.Count}");

            var scorePath = Environment.OSVersion.Platform == PlatformID.Unix ? 
                Path.Combine(".", "images", "score") : @"..\..\..\images\score";
            var scores = Directory.GetFiles(scorePath);

            Console.WriteLine($"Processing {scores.Length} images");
            
            foreach (var score in scores)
            {
                using var theImage = Image.Load<Rgb24>(score);
                var embedding = GetEmbedding(theImage);
                var proto = embeddings.FromSimilarity(embedding);
                var label = proto.Item1;
                var similarity = proto.Item2;
                var filename = Path.GetFileName(score);

                Console.WriteLine($"Image: [{filename}] --> classified as [{label}] with similarity [{similarity}]");
            }

            faceDetector.Dispose();
            _faceLandmarksExtractor.Dispose();
            _faceEmbedder.Dispose();

            Console.WriteLine("Done.");
            Console.ReadKey();
        }

        static float[] GetEmbedding(Image<Rgb24> image)
        {
            var array = GetImageFloatArray(image);
            var rectangles = faceDetector.Forward(array);
            var rectangle = rectangles.FirstOrDefault().Box;

            if (!rectangle.IsEmpty)
            {
                // landmarks
                var points = _faceLandmarksExtractor.Forward(array, rectangle);
                var angle = points.RotationAngle;

                // alignment
                var aligned = FaceProcessingExtensions.Align(array, rectangle, angle);
                return _faceEmbedder.Forward(aligned);
            }

            return new float[512];
        }

        static float[][,] GetImageFloatArray(Image<Rgb24> image)
        {
            var array = new[]
            {
                new float [image.Height,image.Width],
                new float [image.Height,image.Width],
                new float [image.Height,image.Width]
            };

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (var y = 0; y < pixelAccessor.Height; y++)
                {
                    var row = pixelAccessor.GetRowSpan(y);
                    for (var x = 0; x < pixelAccessor.Width; x++)
                    {
                        array[2][y, x] = row[x].R / 255.0F;
                        array[1][y, x] = row[x].G / 255.0F;
                        array[0][y, x] = row[x].B / 255.0F;
                    }
                }
            });

            return array;
        }
    }
}
