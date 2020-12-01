using FaceONNX;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Drawing;

namespace GPUPerfomanceTests
{
    class Program
    {
        const int oneSecond = 1000;
        const int iterations = 1000;
        const int gpuId = 0;

        static void Main()
        {
            // Session options
            Console.WriteLine($"FaceONNX: GPU Perfomance tests with CUDA provider\n");
            using var bitmap = new Bitmap(@"..\..\..\images\brad.jpg");
            var options = SessionOptions.MakeSessionOptionWithCudaProvider(gpuId);

            // FPS tests
            FaceDetectorFPSTest(options, bitmap);
            FaceDetectorLightFPSTest(options, bitmap);
        }

        #region FPS tests
        static void FaceDetectorFPSTest(SessionOptions options, Bitmap bitmap)
        {
            int tic, toc, time;
            float average;
            var faceDetector = new FaceDetector(options);
            Console.WriteLine($"FPS test for [{faceDetector}]");
            Console.WriteLine($"Initializing GPU device [{gpuId}]");
            tic = Environment.TickCount;
            _ = faceDetector.Forward(bitmap);
            toc = Environment.TickCount - tic;
            Console.WriteLine($"Finished in [{toc}] mls.");

            time = 0;
            Console.WriteLine($"Running FPS test for [{iterations}] iterations");

            for (int i = 0; i < iterations; i++)
            {
                tic = Environment.TickCount;
                _ = faceDetector.Forward(bitmap);
                toc = Environment.TickCount - tic;
                time += toc;
            }

            average = time / (float)iterations;
            Console.WriteLine($"FPS --> [{oneSecond / average}]\n");
        }

        static void FaceDetectorLightFPSTest(SessionOptions options, Bitmap bitmap)
        {
            int tic, toc, time;
            float average;
            var faceDetectorLight = new FaceDetectorLight(options);
            Console.WriteLine($"FPS test for [{faceDetectorLight}]");
            Console.WriteLine($"Initializing GPU device [{gpuId}]");
            tic = Environment.TickCount;
            _ = faceDetectorLight.Forward(bitmap);
            toc = Environment.TickCount - tic;
            Console.WriteLine($"Finished in [{toc}] mls.");

            time = 0;
            Console.WriteLine($"Running FPS test for [{iterations}] iterations");

            for (int i = 0; i < iterations; i++)
            {
                tic = Environment.TickCount;
                _ = faceDetectorLight.Forward(bitmap);
                toc = Environment.TickCount - tic;
                time += toc;
            }

            average = time / (float)iterations;
            Console.WriteLine($"FPS --> [{oneSecond / average}]\n");

            Console.WriteLine("Done.");
            Console.ReadKey();
        }
        #endregion
    }
}
