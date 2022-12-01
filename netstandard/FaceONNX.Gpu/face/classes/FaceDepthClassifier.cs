using FaceONNX.Properties;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Defines face depth classifier.
    /// </summary>
    public class FaceDepthClassifier : IFaceClassifier
    {
        #region Private data
        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;
        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face depth classifier.
        /// </summary>
        public FaceDepthClassifier()
        {
            _session = new InferenceSession(Resources.depth_googlenet_slim);
        }

        /// <summary>
        /// Initializes face depth classifier.
        /// </summary>
        /// <param name="options">Session options</param>
        public FaceDepthClassifier(SessionOptions options)
        {
            _session = new InferenceSession(Resources.depth_googlenet_slim, options);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the labels.
        /// </summary>
        public static string[] Labels = new string[] { "Fake", "Real" };

        #endregion

        #region Methods

        /// <inheritdoc/>
        public float[] Forward(Bitmap image)
        {
            var size = new Size(224, 224);
            using var clone = image.Resize(size);
            int width = clone.Width;
            int height = clone.Height;
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 1, height, width };
            var bmData = clone.LockBits(new Rectangle(0, 0, clone.Width, clone.Height),
                ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var tensors = bmData.ToFloatTensor(false);
            tensors.Compute(127.0f, Matrice.Sub);
            var inputData = tensors.Average();
            clone.Unlock(bmData);

            // session run
            var t = new DenseTensor<float>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
            using var outputs = _session.Run(inputs);
            var results = outputs.ToArray();
            var length = results.Length;
            var confidences = results[length - 1].AsTensor<float>().ToArray();

            return confidences;
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

        ~FaceDepthClassifier()
        {
            Dispose(false);
        }

        #endregion
    }
}
