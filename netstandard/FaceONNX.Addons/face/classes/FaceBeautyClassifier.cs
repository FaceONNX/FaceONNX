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
	/// Defines face beauty classifier.
	/// </summary>
	public class FaceBeautyClassifier : IFaceClassifier
    {
		#region Private data
		/// <summary>
		/// Inference session.
		/// </summary>
		private readonly InferenceSession _session;
		#endregion

		#region Constructor

		/// <summary>
		/// Initializes face beauty classifier.
		/// </summary>
		public FaceBeautyClassifier()
		{
			_session = new InferenceSession(Resources.beauty_resnet18);
		}

		/// <summary>
		/// Initializes face beauty classifier.
		/// </summary>
		/// <param name="options">Session options</param>
		public FaceBeautyClassifier(SessionOptions options)
		{
			_session = new InferenceSession(Resources.beauty_resnet18, options);
		}

		#endregion

		#region Methods

		/// <inheritdoc/>
		public float[] Forward(Bitmap image)
		{
            var rgb = image.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public float[] Forward(float[][,] image)
        {
            if (image.Length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var size = new Size(224, 224);
            var resized = new float[3][,];

            for (int i = 0; i < image.Length; i++)
            {
                resized[i] = image[i].Resize(size.Height, size.Width);
            }

            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 3, size.Height, size.Width };
            var tensors = resized.ToFloatTensor(false);
            tensors.Compute(new float[] { 104, 117, 123 }, Matrice.Sub);
            tensors.Compute(255, Matrice.Div);
            var inputData = tensors.Merge(true);

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

		~FaceBeautyClassifier()
		{
			Dispose(false);
		}

		#endregion
	}
}
