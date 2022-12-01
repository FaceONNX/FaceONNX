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
            var size = new Size(112, 112);
            using var clone = image.Resize(size);
            int width = clone.Width;
            int height = clone.Height;
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 3, height, width };
            var bmData = clone.LockBits(new Rectangle(0, 0, clone.Width, clone.Height),
                ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var tensors = bmData.ToFloatTensor(true);
            tensors.Compute(255.0f, Matrice.Div);
            var inputData = tensors.Merge(true);
            clone.Unlock(bmData);

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
                    (int)(confidences[i + 0] * image.Width),
                    (int)(confidences[i + 1] * image.Height));
            }

            return points;
        }

        #endregion

        #region Static methods

        /// <summary>
        /// Returns aligned points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="angle">Angle</param>
        /// <param name="imageSize">Image size</param>
        /// <returns>Points</returns>
        public static Point[] Align(Size imageSize, Point[] points, float angle)
        {
            // rotate points
            var rotated_points = points.Rotate(
                new Point
                {
                    X = imageSize.Width / 2,
                    Y = imageSize.Height / 2
                }, angle);

            return rotated_points;
        }

        /// <summary>
        /// Returns aligned rectangle.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="angle">Angle</param>
        /// <param name="imageSize">Image size</param>
        /// <returns>Rectangle</returns>
        public static Rectangle Align(Size imageSize, Rectangle rectangle, float angle)
        {
            // rotate rectangle points
            var rectangle_rotated_points = rectangle.ToPoints().Rotate(new Point
            {
                X = imageSize.Width / 2,
                Y = imageSize.Height / 2
            }, angle);

            // get mean point
            var mean_point = Landmarks.GetMeanPoint(rectangle_rotated_points);

            // inverse transform
            var rectangle_rotated_points_inverted = rectangle_rotated_points.Rotate(new Point
            {
                X = mean_point.X,
                Y = mean_point.Y
            }, -angle);

            return rectangle_rotated_points_inverted.FromPoints();
        }

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
        /// Returns rotation angle from points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Angle</returns>
        public static float GetRotationAngle(Point[] points)
        {
            var left = Landmarks.GetMeanPoint(points.GetLeftEye());
            var right = Landmarks.GetMeanPoint(points.GetRightEye());
            var point = left.GetSupportedPoint(right);
            var angle = left.GetAngle(right, point);

            return angle;
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

        ~FaceLandmarksExtractor()
        {
            Dispose(false);
        }

        #endregion
    }
}
