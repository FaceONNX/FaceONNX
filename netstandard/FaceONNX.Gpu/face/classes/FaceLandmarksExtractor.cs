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
			var size = new Size(112, 112);
			using var clone = BitmapTransform.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, 3, height, width };
			var tensors = clone.ToFloatTensor(true);
			tensors.Compute(255.0f, Matrice.Div);
            var inputData = tensors.Merge(true);

			// session run
			var t = new DenseTensor<float>(inputData, dimentions);
			var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
			var results = _session.Run(inputs).ToArray();
			var length = results.Length;
			var confidences = results[length - 1].AsTensor<float>().ToArray();
			var points = new Point[confidences.Length / 2];

			for (int i = 0, j = 0; i < (length = confidences.Length); i += 2)
			{
				points[j++] = new Point(
					(int)(confidences[i + 0] * image.Width),
					(int)(confidences[i + 1] * image.Height));
			}

			// dispose
			foreach (var result in results)
			{
				result.Dispose();
			}

			return points;
		}

		#endregion

		#region Static methods

		/// <summary>
		/// Returns aligned face.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <param name="points">Points</param>
		/// <returns>Bitmap</returns>
		public static Bitmap Align(Bitmap image, Point[] points)
		{
			var angle = GetRotationAngle(points);
			return Align(image, angle);
		}

		/// <summary>
		/// Returns aligned face.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <param name="angle">Angle</param>
		/// <returns>Bitmap</returns>
		public static Bitmap Align(Bitmap image, float angle)
		{
			return BitmapTransform.Rotate(image, angle, Color.Black);
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
			var point = Landmarks.GetSupportedPoint(left, right);
			var angle = Landmarks.GetAngle(left, right, point);

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
