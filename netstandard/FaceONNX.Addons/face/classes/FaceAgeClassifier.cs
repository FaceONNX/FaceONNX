using FaceONNX.Addons.Properties;
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
    /// Defines face age classifier.
    /// </summary>
    public class FaceAgeClassifier : IFaceClassifier
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
		public FaceAgeClassifier()
		{
			_session = new InferenceSession(Resources.age_googlenet);
		}

		/// <summary>
		/// Initializes face age classifier.
		/// </summary>
		/// <param name="options">Session options</param>
		public FaceAgeClassifier(SessionOptions options)
		{
			_session = new InferenceSession(Resources.age_googlenet, options);
		}

        #endregion

        #region Properties

        /// <summary>
        /// Returns the labels.
        /// </summary>
        public static string[] Labels = new string[] { "<2", "3-7", "8-14", "15-24", "25-37", "38-47", "48-59", ">60" };

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
			var dimentions = new int[] { 1, 3, height, width };
			var tensors = clone.ToFloatTensor(false);
			tensors.Compute(new float[] { 104, 117, 123 }, Matrice.Sub);
			var inputData = tensors.Merge(true);

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

		~FaceAgeClassifier()
		{
			Dispose(false);
		}

		#endregion
	}
}
