using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace FaceEmbeddingsClassification
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face embeddings classification");
            var fits = Directory.GetFiles(@"..\..\..\images\fit");
            var faceDetectorLight = new FaceDetectorLight(0.75f, 0.25f);
            var faceEmbedder = new FaceEmbedder();
            var embeddings = new Embeddings(0.35f);

            foreach (var fit in fits)
            {
                using var bitmap = new Bitmap(fit);
                var face = faceDetectorLight.Forward(bitmap);
                var embedding = faceEmbedder.Forward(bitmap, face);
                var name = Path.GetFileNameWithoutExtension(fit);
                embeddings.Add(embedding.First(), name);
                Directory.CreateDirectory(name);
            }

            Console.WriteLine($"Embeddings count: {embeddings.Count}");

            var files = Directory.GetFiles(@"..\..\..\images\score");
            Console.WriteLine($"Processing {files.Length} images");

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var face = faceDetectorLight.Forward(bitmap);

                if (face.Length > 0)
                {
                    var embedding = faceEmbedder.Forward(bitmap, face);
                    var proto = embeddings.FromSimilarity(embedding.First());

                    var filename = Path.GetFileName(file);
                    bitmap.Save(Path.Combine(proto, filename));
                    Console.WriteLine($"Image: {filename} --> classified as {proto}");
                }
            }

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
    }
}
