using FaceONNX.Properties;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Xml.Linq;
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
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public FaceDetector(float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.face_detector_640);
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        /// <summary>
        /// Initializes face detector.
        /// </summary>
        /// <param name="options">Session options</param>
        /// <param name="confidenceThreshold">Confidence threshold</param>
        /// <param name="nmsThreshold">NonMaxSuppression threshold</param>
        public FaceDetector(SessionOptions options, float confidenceThreshold = 0.95f, float nmsThreshold = 0.5f)
        {
            _session = new InferenceSession(Resources.face_detector_640, options);
            ConfidenceThreshold = confidenceThreshold;
            NmsThreshold = nmsThreshold;
        }

        #endregion

        #region Properties

        /// <inheritdoc/>
        public float ConfidenceThreshold { get; set; }

        /// <inheritdoc/>
        public float NmsThreshold { get; set; }

        #endregion

        #region Methods

        /// <inheritdoc/>
        public Rectangle[] Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public Rectangle[] Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var width = image[0].GetLength(1);
            var height = image[0].GetLength(0);

            var size = new Size(640, 480);
            var resized = new float[3][,];

            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = image[i].Resize(size.Height, size.Width);
            }

            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 3, size.Height, size.Width };
            var tensors = resized.ToFloatTensor(true);
            tensors.Compute(new float[] { 127.0f, 127.0f, 127.0f }, Matrice.Sub);
            tensors.Compute(128, Matrice.Div);
            var inputData = tensors.Merge(true);

            // session run
            var t = new DenseTensor<float>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
            using var outputs = _session.Run(inputs);
            var results = outputs.ToArray();
            var confidences = results[0].AsTensor<float>().ToArray();
            var boxes = results[1].AsTensor<float>().ToArray();
            var length = confidences.Length;

            // post-proccessing
            var boxes_picked = new List<Rectangle>();

            for (int i = 0, j = 0; i < length; i += 2, j += 4)
            {
                if (confidences[i + 1] > ConfidenceThreshold)
                {
                    boxes_picked.Add(
                        Rectangle.FromLTRB
                            (
                                (int)(boxes[j + 0] * width),
                                (int)(boxes[j + 1] * height),
                                (int)(boxes[j + 2] * width),
                                (int)(boxes[j + 3] * height)
                            ).ToBox());
                }
            }

            // non-max suppression
            length = boxes_picked.Count;

            for (int i = 0; i < length; i++)
            {
                var first = boxes_picked[i];

                for (int j = i + 1; j < length; j++)
                {
                    var second = boxes_picked[j];
                    var iou = first.IoU(second);

                    if (iou > NmsThreshold)
                    {
                        boxes_picked.RemoveAt(j);
                        length = boxes_picked.Count;
                        j--;
                    }
                }
            }

            return boxes_picked.ToArray();
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

        ~FaceDetector()
        {
            Dispose(false);
        }

        #endregion
    }
}
