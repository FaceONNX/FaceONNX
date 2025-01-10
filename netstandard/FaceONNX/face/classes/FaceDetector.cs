using FaceONNX.Properties;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Defines face detector.
    /// </summary>
    public class FaceDetector : IFaceDetector
    {
        #region Private data

        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face detector.
        /// </summary>
        /// <param name="detectionThreshold">Detection threshold</param>
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public FaceDetector(float detectionThreshold = 0.3f, float confidenceThreshold = 0.4f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.yolov5s_face);
            DetectionThreshold = detectionThreshold;
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        /// <summary>
        /// Initializes face detector.
        /// </summary>
        /// <param name="options">Session options</param>
        /// <param name="detectionThreshold">Detection threshold</param>
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public FaceDetector(SessionOptions options, float detectionThreshold = 0.3f, float confidenceThreshold = 0.4f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.yolov5s_face, options);
            DetectionThreshold = detectionThreshold;
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        #endregion

        #region Properties

        /// <inheritdoc/>
        public float DetectionThreshold { get; set; }

        /// <inheritdoc/>
        public float ConfidenceThreshold { get; set; }

        /// <inheritdoc/>
        public float NmsThreshold { get; set; }

        /// <summary>
        /// Gets labels.
        /// </summary>
        public static readonly string[] Labels = new string[]
        {
            "Face"
        };

        #endregion

        #region Methods

        /// <inheritdoc/>
        public FaceDetectionResult[] Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public FaceDetectionResult[] Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            // params
            var width = image[0].GetLength(1);
            var height = image[0].GetLength(0);
            var size = new Size(640, 640);
            var resized = new float[3][,];

            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = image[i].ResizePreserved(size.Height, size.Width, 0.0f, InterpolationMode.Bilinear);
            }

            // yolo params
            var yoloSquare = 15;
            var classes = Labels.Length;
            var count = classes + yoloSquare;

            // pre-processing
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];
            var dimentions = new[] { 1, 3, size.Height, size.Width };
            var tensor = resized.ToFloatTensor(true);
            tensor.Compute(255.0f, Matrice.Div); // scale
            var inputData = tensor.Merge(true);

            // session run
            var t = new DenseTensor<float>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(name, t) };
            using var sessionResults = _session?.Run(inputs);
            var results = sessionResults?.ToArray();

            if (results == null)
                return new FaceDetectionResult[] { };

            // post-processing
            var vector = results[0].AsTensor<float>().ToArray();
            var length = vector.Length / count;
            var predictions = new float[length][];

            for (int i = 0; i < length; i++)
            {
                var prediction = new float[count];

                for (int j = 0; j < count; j++)
                    prediction[j] = vector[i * count + j];

                predictions[i] = prediction;
            }

            var list = new List<float[]>();

            // seivining results
            for (int i = 0; i < length; i++)
            {
                var prediction = predictions[i];

                if (prediction[4] > DetectionThreshold)
                {
                    var a = prediction[0];
                    var b = prediction[1];
                    var c = prediction[2];
                    var d = prediction[3];

                    prediction[0] = a - c / 2;
                    prediction[1] = b - d / 2;
                    prediction[2] = a + c / 2;
                    prediction[3] = b + d / 2;

                    //for (int j = yoloSquare; j < prediction.Length; j++)
                    //{
                    //    prediction[j] *= prediction[4];
                    //}

                    list.Add(prediction);
                }
            }

            // non-max suppression
            list = NonMaxSuppressionExensions.AgnosticNMSFiltration(list, NmsThreshold);

            // perform
            predictions = list.ToArray();
            length = predictions.Length;

            // backward transform
            var k0 = (float)size.Width / width;
            var k1 = (float)size.Height / height;
            float gain = Math.Min(k0, k1);
            float p0 = (size.Width - width * gain) / 2;
            float p1 = (size.Height - height * gain) / 2;

            // collect results
            var detectionResults = new List<FaceDetectionResult>();

            for (int i = 0; i < length; i++)
            {
                var prediction = predictions[i];
                var labels = new float[classes];

                for (int j = 0; j < classes; j++)
                {
                    labels[j] = prediction[j + yoloSquare];
                }

                var max = Matrice.Max(labels, out int argmax);

                if (max > ConfidenceThreshold)
                {
                    var rectangle = Rectangle.FromLTRB(
                        (int)((prediction[0] - p0) / gain),
                        (int)((prediction[1] - p1) / gain),
                        (int)((prediction[2] - p0) / gain),
                        (int)((prediction[3] - p1) / gain));

                    var points = new Point[5];

                    for (int j = 0; j < 5; j++)
                    {
                        points[j] = new Point
                        {
                            X = (int)((prediction[5 + 2 * j + 0] - p0) / gain),
                            Y = (int)((prediction[5 + 2 * j + 1] - p1) / gain)
                        };
                    }

                    var landmarks = new Face5Landmarks(points);

                    detectionResults.Add(new FaceDetectionResult
                    {
                        Rectangle = rectangle,
                        Id = argmax,
                        Score = max,
                        Points = landmarks
                    });
                }
            }

            return detectionResults.ToArray();
        }

        #endregion

        #region IDisposable

        private bool _disposed;

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <inheritdoc/>
        protected void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Destructor.
        /// </summary>
        ~FaceDetector()
        {
            Dispose(false);
        }

        #endregion
    }
}
