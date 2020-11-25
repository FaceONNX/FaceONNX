using System;

namespace FaceONNX.Core
{
    /// <summary>
    /// Using for vector operations.
    /// </summary>
    public static class Vector
    {
        #region Abs function
        /// <summary>
        /// Returns vector module.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <param name="squared">Squared or not</param>
        /// <returns>Value</returns>
        public static float Abs(this float[] vector, bool squared = false)
        {
            int length = vector.Length;
            float r = 0;

            for (int i = 0; i < length; i++)
            {
                r += vector[i] * vector[i];
            }

            if (squared)
                return r;

            return (float)Math.Sqrt(r);
        }
        /// <summary>
        /// Returns matrix module.
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <param name="squared">Squared or not</param>
        /// <returns>Vector</returns>
        public static float[] Abs(this float[][] matrix, bool squared = false)
        {
            int length = matrix.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Abs(matrix[i], squared);
            }

            return c;
        }
        #endregion

        #region Summary function
        /// <summary>
        /// Returns sum of vector.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <returns>Value</returns>
        public static float Sum(this float[] vector)
        {
            int length = vector.Length;
            float r = 0;

            for (int i = 0; i < length; i++)
            {
                r += vector[i];
            }

            return r;
        }
        /// <summary>
        /// Returns sum of matrix.
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Sum(this float[][] matrix)
        {
            int length = matrix.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Sum(matrix[i]);
            }

            return c;
        }
        #endregion

        #region Add/Sub functions
        /// <summary>
        /// Returns summary of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Value</returns>
        public static float[] Add(this float[] a, float[] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] + b[i];
            }
            return c;
        }
        /// <summary>
        /// Returns summary of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Matrix</returns>
        public static float[][] Add(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[][] c = new float[length][];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Add(a[i], b[i]);
            }

            return c;
        }

        /// <summary>
        /// Returns difference of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Sub(this float[] a, float[] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] - b[i];
            }
            return c;
        }
        /// <summary>
        /// Returns difference of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Matrix</returns>
        public static float[][] Sub(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[][] c = new float[length][];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Sub(a[i], b[i]);
            }

            return c;
        }
        #endregion

        #region Error/Loss/similarity functions
        /// <summary>
        /// Returns loss function of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Value</returns>
        public static float Loss(this float[] a, float[] b)
        {
            int length = a.Length;
            float s = 0;

            for (int i = 0; i < length; i++)
            {
                s += Math.Abs(a[i] - b[i]);
            }

            return s;
        }
        /// <summary>
        /// Returns loss function of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Loss(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Loss(a[i], b[i]);
            }

            return c;
        }
        /// <summary>
        /// Returns accuracy function of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Value</returns>
        public static float Accuracy(this float[] a, float[] b)
        {
            float c = Vector.Abs(a) / Vector.Abs(b);

            if (c > 1.0)
                return (float)1.0 / c;

            return c;
        }
        /// <summary>
        /// Returns accuracy function of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Accuracy(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Accuracy(a[i], b[i]);
            }

            return c;
        }
        /// <summary>
        /// Returns similarity function of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Value</returns>
        public static float Similarity(this float[] a, float[] b)
        {
            int length = a.Length;
            float A = Vector.Abs(a);
            float B = Vector.Abs(b);
            float s = 0;

            for (int i = 0; i < length; i++)
                s += a[i] * b[i];

            return s / (A * B);
        }
        /// <summary>
        /// Returns similarity function of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Similarity(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Similarity(a[i], b[i]);
            }

            return c;
        }
        #endregion

        #region Mean function
        /// <summary>
        /// Returns mean vector of two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Mean(this float[] a, float[] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = (a[i] + b[i]) / 2.0f;
            }

            return c;
        }
        /// <summary>
        /// Returns mean matrix of two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Vector</returns>
        public static float[][] Mean(this float[][] a, float[][] b)
        {
            int length = a.Length;
            float[][] c = new float[length][];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Mean(a[i], b[i]);
            }

            return c;
        }
        /// <summary>
        /// Returns mean value of vector.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <returns>Value</returns>
        public static float Mean(this float[] vector)
        {
            return Vector.Sum(vector) / (float)vector.Length;
        }
        /// <summary>
        /// Returns mean value of matrix.
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Mean(this float[][] matrix)
        {
            int length = matrix.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Mean(matrix[i]);
            }

            return c;
        }
        #endregion

        #region Distance functions
        /// <summary>
        /// Returns distance between two vectors.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Vector</param>
        /// <returns>Value</returns>
        public static float Distance(this float[] a, float[] b)
        {
            float[] c = Vector.Sub(a, b);
            return Vector.Abs(c);
        }
        /// <summary>
        /// Returns distance between two matrices.
        /// </summary>
        /// <param name="a">Matrix</param>
        /// <param name="b">Matrix</param>
        /// <returns>Vector</returns>
        public static float[] Distance(float[][] a, float[][] b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Distance(a[i], b[i]);
            }

            return c;
        }
        /// <summary>
        /// Returns distances between matrix and vector.
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <param name="vector">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Distance(this float[][] matrix, float[] vector)
        {
            int length = matrix.GetLength(0);
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = Vector.Distance(matrix[i], vector);
            }

            return c;
        }
        #endregion

        #region Resize function
        /// <summary>
        /// Resizes a vector.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <param name="length">Length</param>
        /// <returns>Value</returns>
        public static float[] Resize(this float[] vector, int length)
        {
            int r0 = Math.Min(vector.GetLength(0), length);
            float[] c = new float[length];

            for (int i = 0; i < r0; i++)
                c[i] = vector[i];

            return c;
        }
        /// <summary>
        /// Resizes a matrix.
        /// </summary>
        /// <param name="matrix">Matrix</param>
        /// <param name="length">Length</param>
        /// <returns>Value</returns>
        public static float[][] Resize(this float[][] matrix, int length)
        {
            int length0 = matrix.Length;
            float[][] c = new float[length0][];

            for (int i = 0; i < length0; i++)
            {
                c[i] = Vector.Resize(matrix[i], length);
            }

            return c;
        }
        #endregion

        #region Solver
        /// <summary>
        /// Returns a vector corresponding to the solution of a system of linear algebraic equations: "Ax = b".
        /// </summary>
        /// <param name="A">Square matrix</param>
        /// <param name="b">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Solve(this float[][] A, float[] b)
        {
            int height = A.GetLength(0);
            int width = A[0].GetLength(0);

            if (height != width)
                throw new Exception("The matrix must be square");
            if (height != b.Length)
                throw new Exception("Vector length should be equal to the height of the matrix");

            float[][] B = new float[height][];
            int i, j, k, l;
            float[] x = (float[])b.Clone();
            float[] v, w;
            float temp;

            for (i = 0; i < height; i++)
            {
                B[i] = new float[width];

                for (j = 0; j < width; j++)
                    B[i][j] = A[i][j];
            }

            for (i = 0; i < height; i++)
            {
                w = B[i];
                temp = w[i];

                for (j = 0; j < width; j++)
                {
                    w[j] /= temp;
                }
                x[i] /= temp;

                for (k = i + 1; k < height; k++)
                {
                    v = B[k];
                    temp = v[i];

                    for (j = i; j < width; j++)
                    {
                        v[j] = v[j] - w[j] * temp;
                    }

                    x[k] -= x[i] * temp;
                    B[k] = v;
                }
            }

            for (i = 0; i < height; i++)
            {
                l = (height - 1) - i;
                w = B[l];

                for (k = 0; k < l; k++)
                {
                    v = B[k];
                    temp = v[l];

                    for (j = l; j < width; j++)
                    {
                        v[j] = v[j] - w[j] * temp;
                    }

                    x[k] -= x[l] * temp;
                    B[k] = v;
                }

                B[l] = w;
            }

            return x;
        }
        #endregion

        #region Argmax/argmin
        /// <summary>
        /// Returns argmax of vector.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <returns>Value</returns>
        public static int Argmax(this float[] vector)
        {
            long length = vector.Length;
            float max = float.MinValue;
            int index = 0;

            for (int i = 0; i < length; i++)
            {
                if (vector[i] > max)
                {
                    max = vector[i];
                    index = i;
                }
            }

            return index;
        }
        /// <summary>
        /// Returns argmax of vector.
        /// </summary>
        /// <param name="vector">Vector</param>
        /// <returns>Value</returns>
        public static int Argmin(this float[] vector)
        {
            long length = vector.Length;
            float min = float.MaxValue;
            int index = 0;

            for (int i = 0; i < length; i++)
            {
                if (vector[i] < min)
                {
                    min = vector[i];
                    index = i;
                }
            }

            return index;
        }
        #endregion

        #region Numerics
        /// <summary>
        /// Returns summary of vector and numeric.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Value</param>
        /// <returns>Vector</returns>
        public static float[] Add(this float[] a, float b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] + b;
            }
            return c;
        }
        /// <summary>
        /// Returns summary of vector and numeric.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Value</param>
        /// <returns>Vector</returns>
        public static float[] Sub(this float[] a, float b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] - b;
            }
            return c;
        }
        /// <summary>
        /// Returns multiplication of vector and numeric.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Value</param>
        /// <returns>Vector</returns>
        public static float[] Mul(this float[] a, float b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] * b;
            }
            return c;
        }
        /// <summary>
        /// Divides vector by the numeric.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <param name="b">Value</param>
        /// <returns>Vector</returns>
        public static float[] Div(this float[] a, float b)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = a[i] / b;
            }
            return c;
        }
        #endregion

        #region Exp/Log
        /// <summary>
        /// Returns exp of vector.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Exp(this float[] a)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = (float)Math.Exp(a[i]);
            }
            return c;
        }
        /// <summary>
        /// Returns log of vector.
        /// </summary>
        /// <param name="a">Vector</param>
        /// <returns>Vector</returns>
        public static float[] Log(this float[] a)
        {
            int length = a.Length;
            float[] c = new float[length];

            for (int i = 0; i < length; i++)
            {
                c[i] = (float)Math.Exp(a[i]);
            }
            return c;
        }
        #endregion
    }
}
