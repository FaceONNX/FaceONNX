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
    /// Defines face landmarks extractor.
    /// </summary>
    public class FaceLandmarksExtractor : IFaceLandmarksExtractor
    {
        #region Private data
        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;
        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face landmarks extractor.
        /// </summary>
        public FaceLandmarksExtractor()
        {
            _session = new InferenceSession(Resources.landmarks_68_pfld);
        }
        /// <summary>
        /// Initializes face landmarks extractor.
        /// </summary>
        /// <param name="options">Session options</param>
        public FaceLandmarksExtractor(SessionOptions options)
        {
            _session = new InferenceSession(Resources.landmarks_68_pfld, options);
        }

        #endregion

        #region Methods

        /// <inheritdoc/>
        public Point[] Forward(Bitmap image)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public Point[] Forward(Bitmap image, Rectangle rectangle, bool clamp = true)
        {
            var rgb = image.ToRGB(false);
            return Forward(rgb, rectangle);
        }

        /// <inheritdoc/>
        public Point[] Forward(float[][,] image, Rectangle rectangle, bool clamp = true)
        {
            var length = image.Length;
            var cropped = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                cropped[i] = image[i].Crop(
                    rectangle.Y, 
                    rectangle.X, 
                    rectangle.Height, 
                    rectangle.Width);
            }

            return Forward(cropped);
        }

        /// <inheritdoc/>
        public Point[] Forward(float[][,] image)
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
                resized[i] = image[i].Resize(size.Height, size.Width);
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

            return points;
        }

        #endregion

        #region Static methods

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="angle">Angle</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Align(Bitmap image, float angle)
        {
            return image.Rotate(angle);
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="angle">Angle</param>
        /// <param name="clamp">Clamp crop or not</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Align(Bitmap image, Rectangle rectangle, float angle, bool clamp = true)
        {
            var scaledRectangle = rectangle.Scale();
            using var cropped = image.Crop(scaledRectangle, clamp);
            using var aligned = FaceLandmarksExtractor.Align(cropped, angle);
            var cropRectangle = rectangle.Sub(new Point
            {
                X = scaledRectangle.X,
                Y = scaledRectangle.Y
            });

            return aligned.Crop(cropRectangle, clamp);
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="angle">Angle</param>
        /// <returns>Image in BGR terms</returns>
        public static float[][,] Align(float[][,] image, float angle)
        {
            var length = image.Length;

            if (length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var aligned = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                aligned[i] = image[i].Rotate(-angle);
            }

            return aligned;
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="angle">Angle</param>
        /// <param name="clamp">Clamp crop or not</param>
        /// <returns>Image in BGR terms</returns>
        public static float[][,] Align(float[][,] image, Rectangle rectangle, float angle, bool clamp = true)
        {
            var length = image.Length;

            if (length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var scaledRectangle = rectangle.Scale();
            var cropped = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                cropped[i] = image[i].Crop(
                    scaledRectangle.Y, 
                    scaledRectangle.X,
                    scaledRectangle.Height, 
                    scaledRectangle.Width, clamp);
            }

            var aligned = FaceLandmarksExtractor.Align(cropped, angle);
            var cropRectangle = rectangle.Sub(new Point
            {
                X = scaledRectangle.X,
                Y = scaledRectangle.Y
            });

            var output = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                output[i] = aligned[i].Crop(
                    cropRectangle.Y,
                    cropRectangle.X,
                    cropRectangle.Height,
                    cropRectangle.Width, clamp);
            }

            return output;
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
        ~FaceLandmarksExtractor()
        {
            Dispose(false);
        }

        #endregion
    }
}
