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
    /// Defines face 68 landmarks extractor.
    /// </summary>
    public class Face68LandmarksExtractor : IFace68LandmarksExtractor
    {
        #region Private data

        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face 68 landmarks extractor.
        /// </summary>
        public Face68LandmarksExtractor()
        {
            _session = new InferenceSession(Resources.landmarks_68_pfld);
        }
        /// <summary>
        /// Initializes face 68 landmarks extractor.
        /// </summary>
        /// <param name="options">Session options</param>
        public Face68LandmarksExtractor(SessionOptions options)
        {
            _session = new InferenceSession(Resources.landmarks_68_pfld, options);
        }

        #endregion

        #region Methods

        /// <inheritdoc/>
        public Face68Landmarks Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public Face68Landmarks Forward(Bitmap image, Rectangle rectangle, bool clamp = true)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb, rectangle, clamp);
        }

        /// <inheritdoc/>
        public Face68Landmarks Forward(float[][,] image, Rectangle rectangle, bool clamp = true)
        {
            var length = image.Length;
            var cropped = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                cropped[i] = image[i].Crop(
                    rectangle.Y, 
                    rectangle.X, 
                    rectangle.Height, 
                    rectangle.Width,
                    clamp);
            }

            return Forward(cropped);
        }

        /// <inheritdoc/>
        public Face68Landmarks Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            // resize
            var width = image[0].GetLength(1);
            var height = image[0].GetLength(0);
            var size = new Size(112, 112);
            var resized = new float[3][,];

            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = image[i].Resize(size.Height, size.Width, InterpolationMode.Bilinear);
            }

            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 3, size.Height, size.Width };
            var tensors = resized.ToFloatTensor(true);
            tensors.Compute(255.0f, Matrice.Div);
            var inputData = tensors.Merge(true);

            // session run
            var t = new DenseTensor<float>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
            using var outputs = _session.Run(inputs);
            var results = outputs.ToArray();
            var length = results.Length;
            var confidences = results[length - 1].AsTensor<float>().ToArray();
            var points = new Point[confidences.Length / 2];

            for (int i = 0, j = 0; i < (length = confidences.Length); i += 2)
            {
                points[j++] = new Point(
                    (int)(confidences[i + 0] * width),
                    (int)(confidences[i + 1] * height));
            }

            return new Face68Landmarks(points);
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

        private void Dispose(bool disposing)
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
        ~Face68LandmarksExtractor()
        {
            Dispose(false);
        }

        #endregion
    }
}
