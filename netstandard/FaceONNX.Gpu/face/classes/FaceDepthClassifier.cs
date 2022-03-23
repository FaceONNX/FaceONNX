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
		/// Initializes face age classifier.
		/// </summary>
		public FaceDepthClassifier()
		{
			_session = new InferenceSession(Resources.depth_googlenet_slim);
		}

		/// <summary>
		/// Initializes face age classifier.
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
        public float[][] Forward(Bitmap image, params Rectangle[] rectangles)
        {
			int length = rectangles.Length;
			float[][] vector = new float[length][];

			for (int i = 0; i < length; i++)
            {
				var rectangle = rectangles[i];
				using var cropped = BitmapTransform.Crop(image, rectangle);
				vector[i] = Forward(cropped);
            }

			return vector;
        }

		/// <inheritdoc/>
		public float[] Forward(Bitmap image)
		{
			var size = new Size(224, 224);
			using var clone = BitmapTransform.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, 1, height, width };
			var tensors = clone.ToFloatTensor(false);
			tensors.Compute(127.0f, Matrice.Sub);
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
