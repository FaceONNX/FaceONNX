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
    /// Defines face segmentation parser.
    /// </summary>
    public class FaceParser : IFaceParser, IDisposable
    {
        #region Private data
        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;
        #endregion

        #region Class components
        /// <summary>
        /// Initializes face segmentation parser.
        /// </summary>
        public FaceParser()
        {
            _session = new InferenceSession(Resources.face_unet_512);
        }
        /// <summary>
        /// Initializes face segmentation parser.
        /// </summary>
        /// <param name="options">Session options</param>
        public FaceParser(SessionOptions options)
        {
            _session = new InferenceSession(Resources.face_unet_512, options);
        }
        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image</param>
        /// <param name="rectangles">Rectangles</param>
        /// <returns>Array</returns>
        public float[][][,] Forward(Bitmap image, params Rectangle[] rectangles)
        {
            int length = rectangles.Length;
            float[][][,] vector = new float[length][][,];

            for (int i = 0; i < length; i++)
            {
                var rectangle = rectangles[i];
                using var cropped = BitmapTransform.Crop(image, rectangle);
                vector[i] = Forward(cropped);
            }

            return vector;
        }
        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Array</returns>
        public float[][,] Forward(Bitmap image)
        {
            var size = new Size(512, 512);
            using var clone = BitmapTransform.Resize(image, size);
            int width = clone.Width;
            int height = clone.Height;
            var inputMeta = _session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];

            // pre-processing
            var dimentions = new int[] { 1, 3, height, width };
            var tensors = clone.ToFloatTensor(true);
            tensors.Compute(255.0f, Matrice.Div);
            tensors.Compute(new float[] { 0.5f, 0.5f, 0.5f }, Matrice.Sub);
            tensors.Compute(new float[] { 0.5f, 0.5f, 0.5f }, Matrice.Div);
            var inputData = tensors.Merge(true);

            // session run
            var t = new DenseTensor<float>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, t) };
            var results = _session.Run(inputs).ToArray();
            var length = results.Length;
            var confidences = results[length - 1].AsTensor<float>().ToArray();

            // normalize
            var max = confidences.Max();
            var min = confidences.Min();

            var probabilities = Matrice.Div(Matrice.Sub(confidences, min), max - min);
            var outputSize = 19;
            var array = new float[outputSize][,];

            // post-process
            for (int i = 0, k = 0; i < outputSize; i++)
            {
                array[i] = new float[size.Height, size.Width];

                for (int y = 0; y < size.Height; y++)
                {
                    for (int x = 0; x < size.Width; x++)
                    {
                        array[i][y, x] = confidences[k++];
                    }
                }
            }

            // dispose
            foreach (var result in results)
            {
                result.Dispose();
            }

            return array;
        }
        /// <summary>
        /// Returns bitmap from masks.
        /// </summary>
        /// <param name="masks">Masks</param>
        /// <returns>Bitmap</returns>
        public unsafe static Bitmap ToBitmap(params float[][,] masks)
        {
            int height = masks[0].GetLength(0);
            int width = masks[0].GetLength(1);
            var bitmap = new Bitmap(width, height);
            var bitmapData = BitmapFormat.Lock32bpp(bitmap);

            byte* p = (byte*)bitmapData.Scan0.ToPointer();
            int stride = bitmapData.Stride;
            int length = masks.Length;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int k, ystride = y * stride;
                    k = ystride + x * 3;
                    var max = float.MinValue;
                    var index = 0;

                    for (int i = 0; i < length; i++)
                    {
                        if (masks[i][y, x] > max)
                        {
                            max = masks[i][y, x];
                            index = i;
                        }
                    }

                    // transform
                    var color = Labels[index];
                    p[k + 2] = color.R;
                    p[k + 1] = color.G;
                    p[k + 0] = color.B;
                }
            }

            BitmapFormat.Unlock(bitmap, bitmapData);
            return bitmap;
        }
        /// <summary>
        /// Returns the labels.
        /// </summary>
        public static Color[] Labels = new Color[]
        {
            Color.FromArgb(0, 0, 0),
            Color.FromArgb(204, 0, 0),
            Color.FromArgb(76, 153, 0),
            Color.FromArgb(204, 204, 0),
            Color.FromArgb(51, 51, 255),
            Color.FromArgb(204, 0, 204),
            Color.FromArgb(0, 255, 255),
            Color.FromArgb(51, 255, 255),
            Color.FromArgb(102, 51, 0),
            Color.FromArgb(255, 0, 0),
            Color.FromArgb(102, 204, 0),
            Color.FromArgb(255, 255, 0),
            Color.FromArgb(0, 0, 153),
            Color.FromArgb(0, 0, 204),
            Color.FromArgb(255, 51, 153),
            Color.FromArgb(0, 204, 204),
            Color.FromArgb(0, 51, 0),
            Color.FromArgb(255, 153, 51),
            Color.FromArgb(0, 204, 0)
        };
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
