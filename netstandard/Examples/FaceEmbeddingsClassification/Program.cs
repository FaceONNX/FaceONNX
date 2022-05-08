using FaceONNX;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using UMapx.Imaging;

namespace FaceEmbeddingsClassification
{
    class Program
    {
        static FaceDetector faceDetector;
        static FaceLandmarksExtractor _faceLandmarksExtractor;
        static FaceEmbedder _faceEmbedder;

        static void Main()
        {
            Console.WriteLine("FaceONNX: Face embeddings classification");
            var fits = Directory.GetFiles(@"..\..\..\images\fit", "*.*", SearchOption.AllDirectories);
            faceDetector = new FaceDetector();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceEmbedder = new FaceEmbedder();
            var embeddings = new Embeddings();

            foreach (var fit in fits)
            {
                using var bitmap = new Bitmap(fit);
                var embedding = GetEmbedding(bitmap);
                var name = Path.GetFileNameWithoutExtension(fit);
                embeddings.Add(embedding, name);
            }

            Console.WriteLine($"Embeddings count: {embeddings.Count}");
            var scores = Directory.GetFiles(@"..\..\..\images\score");
            Console.WriteLine($"Processing {scores.Length} images");
            
            foreach (var score in scores)
            {
                var bitmap =new Bitmap(score);
                var embedding = GetEmbedding(bitmap);
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

        static float[] GetEmbedding(Bitmap image)
        {
            var rectangles = faceDetector.Forward(image);
            var rectangle = rectangles.FirstOrDefault();

            if (!rectangle.IsEmpty)
            {
                // landmarks
                using var cropped = BitmapTransform.Crop(image, rectangle);
                var points = _faceLandmarksExtractor.Forward(cropped);
                var angle = FaceLandmarksExtractor.GetRotationAngle(points);

                // new alignment
                using var aligned = FaceLandmarksExtractor.Align(image, angle);
                var aligned_rectangle = FaceLandmarksExtractor.Align(image.Size, rectangle, angle);
                using var aligned_cropped = aligned.Crop(aligned_rectangle);

                // old alignment
                //using var aligned_cropped = FaceLandmarksExtractor.Align(cropped, angle);
                return _faceEmbedder.Forward(aligned_cropped);
            }

            return new float[512];
        }
    }
}
