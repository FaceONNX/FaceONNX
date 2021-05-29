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
    /// Defines face detector.
    /// </summary>
    public class FaceDetector : IFaceDetector, IDisposable
    {
		#region Private data
		/// <summary>
		/// Inference session.
		/// </summary>
		private readonly InferenceSession _session;
		#endregion

		#region Class components
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
		/// <summary>
		/// Gets or sets confidence threshold.
		/// </summary>
		public float ConfidenceThreshold { get; set; }
		/// <summary>
		/// Gets or sets NonMaxSuppression threshold.
		/// </summary>
		public float NmsThreshold { get; set; }
		/// <summary>
		/// Returns face detection results.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <returns>Rectangles</returns>
		public Rectangle[] Forward(Bitmap image)
		{
			var size = new Size(640, 480);
			using var clone = BitmapTransform.Resize(image, size);
			int width = clone.Width;
			int height = clone.Height;
			var inputMeta = _session.InputMetadata;
			var name = inputMeta.Keys.ToArray()[0];

			// pre-processing
			var dimentions = new int[] { 1, 3, height, width };
			var tensors = clone.ToFloatTensor(true);
			tensors.Compute(new float[] { 127.0f, 127.0f, 127.0f }, Matrice.Sub);
			tensors.Compute(128, Matrice.Div);
			var inputData = tensors.Merge(true);

			// session run
			var t = new DenseTensor<float>(inputData, dimentions);
			var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
			var results = _session.Run(inputs).ToArray();
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
						Imaging.ToBox(
							Rectangle.FromLTRB
							(
								(int)(boxes[j + 0] * image.Width),
								(int)(boxes[j + 1] * image.Height),
								(int)(boxes[j + 2] * image.Width),
								(int)(boxes[j + 3] * image.Height)
							)));
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
					var iou = Imaging.IoU(first, second);

					if (iou > NmsThreshold)
					{
						boxes_picked.RemoveAt(j);
						length = boxes_picked.Count;
						j--;
					}
				}
			}

			// dispose
			foreach (var result in results)
			{
				result.Dispose();
			}

			return boxes_picked.ToArray();
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
