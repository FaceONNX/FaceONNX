using System.Drawing;
using System.Drawing.Imaging;

namespace FaceONNX.Core
{
    /// <summary>
    /// Onnx tensor class.
    /// </summary>
    public static class Onnx
    {
        #region ToTensor
        /// <summary>
        /// Converts a Bitmap to an BGR tensor arrays.
        /// </summary>
        /// <param name="Data">Bitmap</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>RGB tensor arrays</returns>
        public static byte[][] ToByteTensor(this Bitmap Data, bool rgb = false)
        {
            BitmapData bmData = Imaging.Lock24bpp(Data);
            byte[][] _ix = Onnx.ToByteTensor(bmData, rgb);
            Imaging.Unlock(Data, bmData);
            return _ix;
        }
        /// <summary>
        /// Converts a Bitmap to an BGR tensor arrays.
        /// </summary>
        /// <param name="bmData">Bitmap data</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>RGB tensor arrays</returns>
        public unsafe static byte[][] ToByteTensor(this BitmapData bmData, bool rgb = false)
        {
            // params
            int width = bmData.Width, height = bmData.Height, stride = bmData.Stride;
            byte* p = (byte*)bmData.Scan0.ToPointer();
            int shift = height * width;
            byte[] _ix0 = new byte[shift];
            byte[] _ix1 = new byte[shift];
            byte[] _ix2 = new byte[shift];
            int z = 0;

            // do job
            if (rgb)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        _ix0[z] = p[k + 2];
                        _ix1[z] = p[k + 1];
                        _ix2[z] = p[k + 0];
                    }
                }
            }
            else
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        _ix0[z] = p[k + 0];
                        _ix1[z] = p[k + 1];
                        _ix2[z] = p[k + 2];
                    }
                }
            }

            // arrays
            return new byte[][] { _ix0, _ix1, _ix2 };
        }
        /// <summary>
        /// Converts a Bitmap to an BGR tensor arrays.
        /// </summary>
        /// <param name="Data">Bitmap</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>RGB tensor arrays</returns>
        public static float[][] ToFloatTensor(this Bitmap Data, bool rgb = false)
        {
            BitmapData bmData = Imaging.Lock24bpp(Data);
            float[][] _ix = Onnx.ToFloatTensor(bmData, rgb);
            Imaging.Unlock(Data, bmData);
            return _ix;
        }
        /// <summary>
        /// Converts a Bitmap to an BGR tensor arrays.
        /// </summary>
        /// <param name="bmData">Bitmap data</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>RGB tensor arrays</returns>
        public unsafe static float[][] ToFloatTensor(this BitmapData bmData, bool rgb = false)
        {
            // params
            int width = bmData.Width, height = bmData.Height, stride = bmData.Stride;
            byte* p = (byte*)bmData.Scan0.ToPointer();
            int shift = height * width;
            float[] _ix0 = new float[shift];
            float[] _ix1 = new float[shift];
            float[] _ix2 = new float[shift];
            int z = 0;

            // do job
            if (rgb)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        _ix0[z] = p[k + 2];
                        _ix1[z] = p[k + 1];
                        _ix2[z] = p[k + 0];
                    }
                }
            }
            else
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        _ix0[z] = p[k + 0];
                        _ix1[z] = p[k + 1];
                        _ix2[z] = p[k + 2];
                    }
                }
            }

            // arrays
            return new float[][] { _ix0, _ix1, _ix2 };
        }
        #endregion

        #region FromTensor
        /// <summary>
        /// Converts a BGR tensor arrays to Bitmap.
        /// </summary>
        /// <param name="tensor">Tensor arrays</param>
        /// <param name="width">Width</param>
        /// <param name="height">Height</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>Bitmap</returns>
        public unsafe static Bitmap FromByteTensor(this byte[][] tensor, int width, int height, bool rgb = false)
        {
            // params
            Bitmap Data = new Bitmap(width, height);
            BitmapData bmData = Imaging.Lock24bpp(Data);
            int stride = bmData.Stride;
            byte* p = (byte*)bmData.Scan0.ToPointer();
            int shift = height * width;
            int z = 0;

            // do job
            if (rgb)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        p[k + 2] = tensor[0][z];
                        p[k + 1] = tensor[1][z];
                        p[k + 0] = tensor[2][z];
                    }
                }
            }
            else
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        p[k + 0] = tensor[0][z];
                        p[k + 1] = tensor[1][z];
                        p[k + 2] = tensor[2][z];
                    }
                }
            }

            // arrays
            Imaging.Unlock(Data, bmData);
            return Data;
        }
        /// <summary>
        /// Converts a BGR tensor arrays to Bitmap.
        /// </summary>
        /// <param name="tensor">Tensor arrays</param>
        /// <param name="width">Width</param>
        /// <param name="height">Height</param>
        /// <param name="rgb">RGB or BGR</param>
        /// <returns>Bitmap</returns>
        public unsafe static Bitmap FromFloatTensor(this float[][] tensor, int width, int height, bool rgb = false)
        {
            // params
            Bitmap Data = new Bitmap(width, height);
            BitmapData bmData = Imaging.Lock24bpp(Data);
            int stride = bmData.Stride;
            byte* p = (byte*)bmData.Scan0.ToPointer();
            int shift = height * width;
            int z = 0;

            // do job
            if (rgb)
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        p[k + 2] = (byte)tensor[0][z];
                        p[k + 1] = (byte)tensor[1][z];
                        p[k + 0] = (byte)tensor[2][z];
                    }
                }
            }
            else
            {
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++, z++)
                    {
                        int k, jstride = j * stride;
                        k = jstride + i * 3;

                        // transform
                        p[k + 0] = (byte)tensor[0][z];
                        p[k + 1] = (byte)tensor[1][z];
                        p[k + 2] = (byte)tensor[2][z];
                    }
                }
            }

            // arrays
            Imaging.Unlock(Data, bmData);
            return Data;
        }
        #endregion

        #region Merge
        /// <summary>
        /// Merges image tensors to single tensor.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <param name="slice">Slice or not</param>
        /// <returns>Byte array</returns>
        public static byte[] Merge(this byte[][] image, bool slice = false)
        {
            int count = image.Length;
            int length = image[0].GetLength(0);
            byte[] _ix = new byte[count * length];
            int z = 0;

            if (slice)
            {
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < length; j++)
                    {
                        _ix[z++] = image[i][j];
                    }
                }
            }
            else
            {
                for (int j = 0; j < length; j++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        _ix[z++] = image[i][j];
                    }
                }
            }

            return _ix;
        }
        /// <summary>
        /// Merges image tensors to single tensor.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <param name="slice">Slice or not</param>
        /// <returns>Float array</returns>
        public static float[] Merge(this float[][] image, bool slice = false)
        {
            int count = image.Length;
            int length = image[0].GetLength(0);
            float[] _ix = new float[count * length];
            int z = 0;

            if (slice)
            {
                for (int i = 0; i < count; i++)
                {
                    for (int j = 0; j < length; j++)
                    {
                        _ix[z++] = image[i][j];
                    }
                }
            }
            else
            {
                for (int j = 0; j < length; j++)
                {
                    for (int i = 0; i < count; i++)
                    {
                        _ix[z++] = image[i][j];
                    }
                }
            }

            return _ix;
        }
        #endregion

        #region Average
        /// <summary>
        /// Averages image tensors to single tensor.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <returns>Byte array</returns>
        public static byte[] Average(this byte[][] image)
        {
            int count = image.Length;
            int length = image[0].GetLength(0);
            byte[] _ix = new byte[length];
            int z = 0;

            for (int j = 0; j < length; j++)
            {
                var value = 0;

                for (int i = 0; i < count; i++)
                {
                     value += image[i][j];
                }

                _ix[z++] = (byte)(value / count);
            }

            return _ix;
        }
        /// <summary>
        /// Averages image tensors to single tensor.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <returns>Byte array</returns>
        public static float[] Average(this float[][] image)
        {
            int count = image.Length;
            int length = image[0].GetLength(0);
            float[] _ix = new float[length];
            int z = 0;

            for (int j = 0; j < length; j++)
            {
                var value = 0.0f;

                for (int i = 0; i < count; i++)
                {
                    value += image[i][j];
                }

                _ix[z++] = (float)(value / count);
            }

            return _ix;
        }
        #endregion

        #region Operator
        /// <summary>
        /// Implements operator function.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <param name="b">Vector</param>
        /// <param name="_operator">Operator</param>
        public static void Operator(this float[][] image, float[] b, IOperator _operator)
        {
            int count = image.Length;

            for (int i = 0; i < count; i++)
            {
                image[i] = _operator(image[i], b[i]);
            }

            return;
        }
        /// <summary>
        /// Implements operator function.
        /// </summary>
        /// <param name="image">RGB tensor arrays</param>
        /// <param name="b">Value</param>
        /// <param name="_operator">Operator</param>
        public static void Operator(this float[][] image, float b, IOperator _operator)
        {
            int count = image.Length;

            for (int i = 0; i < count; i++)
            {
                image[i] = _operator(image[i], b);
            }

            return;
        }
        #endregion

        #region Delegate
        /// <summary>
        /// Operator.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Value</param>
        /// <returns></returns>
        public delegate float[] IOperator(float[] a, float b);
        #endregion
    }
}
