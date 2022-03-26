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
	/// Defines eye blink classifier.
	/// </summary>
	public class EyeBlinkClassifier : IFaceClassifier
	{
		#region Private data

		/// <summary>
		/// Inference session.
		/// </summary>
		private readonly InferenceSession _session;

		#endregion

		#region Constructor

		/// <summary>
		/// Initializes eye blink classifier.
		/// </summary>
		public EyeBlinkClassifier()
		{
			_session = new InferenceSession(Resources.eye_blink_cnn);
		}

		/// <summary>
		/// Initializes eye blink classifier.
		/// </summary>
		/// <param name="options">Session options</param>
		public EyeBlinkClassifier(SessionOptions options)
		{
			_session = new InferenceSession(Resources.eye_blink_cnn, options);
		}

		#endregion

		#region Methods

		/// <inheritdoc/>
		public float[] Forward(Bitmap image)
		{
			var size = new Size(34, 26);
			using var clone = BitmapTransform.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, height, width, 1 };
			var tensors = clone.ToFloatTensor(false);
			tensors.Compute(255.0f, Matrice.Div);
			var inputData = tensors.Average();

			// session run
			var t = new DenseTensor<float>(inputData, dimentions);
			var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
			var results = _session.Run(inputs).ToArray();
			var length = results.Length;
			var confidences = results[length - 1].AsTensor<float>().ToArray();

			// dispose
			foreach (var result in results)
			{
				result.Dispose();
			}

			return confidences;
		}

        #endregion

        #region Static

		/// <summary>
		/// Returns left and right eye rectangles from facelandmarks.
		/// </summary>
		/// <param name="points">Points</param>
		/// <returns>Left and right eye rectangles</returns>
		public static (Rectangle, Rectangle) GetEyesRectangles(Point[] points)
        {
			var factor_y = -0.3f;

			var left_eye_rect = points.GetLeftEye()
				.GetRectangle()
				.ToBox()
				.Scale(0.0f, factor_y);

			var right_eye_rect = points.GetRightEye()
				.GetRectangle()
				.ToBox()
				.Scale(0.0f, factor_y);

			return (left_eye_rect, right_eye_rect);
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

		~EyeBlinkClassifier()
		{
			Dispose(false);
		}

		#endregion
	}
}
