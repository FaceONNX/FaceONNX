using FaceONNX.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace FaceONNX
{
    /// <summary>
    /// Defines face landmarks extractor.
    /// </summary>
    public class FaceLandmarksExtractor : IFaceLandmarksExtractor, IDisposable
	{
		#region Private data
		/// <summary>
		/// Inference session.
		/// </summary>
		private readonly InferenceSession _session;
		#endregion

		#region Class components
		/// <summary>
		/// Initializes face landmarks extractor.
		/// </summary>
		public FaceLandmarksExtractor()
		{
			_session = new InferenceSession(Properties.Resources.landmarks_68_pfld);
		}
		/// <summary>
		/// Initializes face landmarks extractor.
		/// </summary>
		/// <param name="options">Session options</param>
		public FaceLandmarksExtractor(SessionOptions options)
		{
			_session = new InferenceSession(Properties.Resources.landmarks_68_pfld, options);
		}
		/// <summary>
		/// Returns face landmarks.
		/// </summary>
		/// <param name="image">Image</param>
		/// <param name="rectangles">Rectangles</param>
		/// <returns>Points</returns>
		public Point[][] Forward(Bitmap image, params Rectangle[] rectangles)
		{
			var length = rectangles.Length;
			var vector = new Point[length][];

			for (int i = 0; i < length; i++)
			{
				var rectangle = rectangles[i];
				using var cropped = Imaging.Crop(image, rectangle);
				var points = Forward(cropped);
				var count = points.Length;

				for (int j = 0; j < points.Length; j++)
				{
					points[j] = new Point(
						points[j].X + rectangle.X,
						points[j].Y + rectangle.Y);
				}

				vector[i] = points;
			}

			return vector;
		}
		/// <summary>
		/// Returns face landmarks.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <returns>Points</returns>
		public Point[] Forward(Bitmap image)
		{
			var size = new Size(112, 112);
			using var clone = Imaging.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, 3, height, width };
			var tensors = clone.ToFloatTensor(true);
			tensors.Operator(255.0f, Vector.Div);

			// normalizers for mobilenet_se
            //tensors.Operator(new float[] { 0.485f, 0.456f, 0.406f }, Vector.Sub);
            //tensors.Operator(new float[] { 0.229f, 0.224f, 0.225f }, Vector.Div);
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
		/// <summary>
		/// Returns aligned face.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <param name="points">Points</param>
		/// <returns>Bitmap</returns>
		public static Bitmap Align(Bitmap image, Point[] points)
		{
			var left = Landmarks.GetMeanPoint(points.GetLeftEye());
			var right = Landmarks.GetMeanPoint(points.GetRightEye());
			var point = Landmarks.GetSupportedPoint(left, right);
			var angle = Landmarks.GetAngle(left, right, point);
			return Imaging.Rotate(image, angle, Color.Black);
		}
		#endregion

		#region Dispose
		/// <summary>
		/// Disposed or not.
		/// </summary>
		private bool _disposed = false;
		/// <summary>
		/// Dispose void.
		/// </summary>
		public void Dispose()
		{
			if (!_disposed)
			{
				_session.Dispose();
				_disposed = true;
			}
		}
		#endregion
	}
}
