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
    public class FaceEmotionClassifier : IFaceClassifier, IDisposable
	{
		#region Private data
		/// <summary>
		/// Inference session.
		/// </summary>
		private readonly InferenceSession _session;
		#endregion

		#region Class components
		/// <summary>
		/// Initializes face age classifier.
		/// </summary>
		public FaceEmotionClassifier()
		{
			_session = new InferenceSession(Resources.emotion_cnn);
		}
		/// <summary>
		/// Initializes face age classifier.
		/// </summary>
		/// <param name="options">Session options</param>
		public FaceEmotionClassifier(SessionOptions options)
		{
			_session = new InferenceSession(Resources.emotion_cnn, options);
		}
		/// <summary>
		/// Returns the labels.
		/// </summary>
		public static string[] Labels = new string[] { "Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disguest", "Fear" };
		/// <summary>
		/// Returns face recognition results.
		/// </summary>
		/// <param name="image">Image</param>
		/// <param name="rectangles">Rectangles</param>
		/// <returns>Array</returns>
		public float[][] Forward(Bitmap image, params Rectangle[] rectangles)
		{
			int length = rectangles.Length;
			float[][] vector = new float[length][];

			for (int i = 0; i < length; i++)
			{
				using var cropped = BitmapTransform.Crop(image, rectangles[i]);
				vector[i] = Forward(cropped);
			}

			return vector;
		}
		/// <summary>
		/// Returns face recognition results.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <returns>Array</returns>
		public float[] Forward(Bitmap image)
		{
			var size = new Size(48, 48);
			using var clone = BitmapTransform.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, 1, height, width };
			var tensors = clone.ToFloatTensor(true);
			tensors.Compute(256, Matrice.Div);
			var inputData = tensors.Average();

			// session run
			var t = new DenseTensor<float>(inputData, dimentions);
			var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
			var results = _session.Run(inputs).ToArray();
			var length = results.Length;
			var confidences = Matrice.Compute(results[length - 1].AsTensor<float>().ToArray(), Maths.Exp);

			// dispose
			foreach (var result in results)
			{
				result.Dispose();
			}

			return confidences;
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
