using FaceONNX;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Drawing;

namespace GPUPerfomanceTests
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine($"FaceONNX: GPU Perfomance tests with CUDA provider");
            using var bitmap = new Bitmap(@"..\..\..\images\brad.jpg");

            var iterations = 100;

            FaceRecognitionTest(bitmap, true, iterations);
            FaceRecognitionTest(bitmap, false, iterations);
        }

        static void FaceRecognitionTest(Bitmap bitmap, bool useGPU, int iterations)
        {
            Console.WriteLine($"{Environment.NewLine}Configuring {nameof(FaceRecognitionTest)}");

            var gpuId = 0;
            var oneSecond = 1000;
            var time = 0;
            var tic = Environment.TickCount;

            using var options = useGPU ? SessionOptions.MakeSessionOptionWithCudaProvider(gpuId) : new SessionOptions();
            Console.WriteLine($"Configuring {(useGPU ? "GPU" : "CPU")} device");

            using var faceLandmarksExtractor = new FaceLandmarksExtractor(options);
            using var faceEmbedder = new FaceEmbedder(options);

            var toc = Environment.TickCount - tic;
            Console.WriteLine($"Finished in [{toc}] ms");

            var average = default(float);
            Console.WriteLine($"Running test for [{iterations}] iterations");

            for (int i = 0; i < iterations; i++)
            {
                tic = Environment.TickCount;

                var points = faceLandmarksExtractor.Forward(bitmap);
                var angle = points.GetRotationAngle();
                using var aligned = FaceLandmarksExtractor.Align(bitmap, angle);
                var embeddings = faceEmbedder.Forward(aligned);

                toc = Environment.TickCount - tic;

                if (i > 0) time += toc;
            }

            average = time / (float)iterations;

            Console.WriteLine($"Average time --> [{average}] ms");
            Console.WriteLine($"FPS --> [{oneSecond / average}]");
            Console.WriteLine($"Finished in [{time}] ms{Environment.NewLine}");
        }

    }
}
