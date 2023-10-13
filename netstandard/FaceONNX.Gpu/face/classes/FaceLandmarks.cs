using System;
using System.Drawing;
using UMapx.Core;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Using for face points operations.
    /// </summary>
    public static class FaceLandmarks
    {
        #region Face operators

        /// <summary>
        /// Returns right eye points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetRightEye(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var eye = new Point[6];

            for (int i = 0; i < 6; i++)
            {
                eye[i] = points[i + 42];
            }

            return eye;
        }

        /// <summary>
        /// Returns left eye points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetLeftEye(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var eye = new Point[6];

            for (int i = 0; i < 6; i++)
            {
                eye[i] = points[i + 36];
            }

            return eye;
        }

        /// <summary>
        /// Returns mouth points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetMouth(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var tongue = new Point[17];

            for (int i = 0; i < 17; i++)
            {
                tongue[i] = points[i + 48];
            }

            return tongue;
        }

        /// <summary>
        /// Returns face points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetFace(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var tongue = new Point[17];

            for (int i = 0; i < 17; i++)
            {
                tongue[i] = points[i];
            }

            return tongue;
        }

        /// <summary>
        /// Returns left brow points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetLeftBrow(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var eye = new Point[5];

            for (int i = 0; i < 5; i++)
            {
                eye[i] = points[i + 17];
            }

            return eye;
        }

        /// <summary>
        /// Returns right brow points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetRightBrow(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var eye = new Point[5];

            for (int i = 0; i < 5; i++)
            {
                eye[i] = points[i + 22];
            }

            return eye;
        }

        /// <summary>
        /// Returns nose points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Points</returns>
        public static Point[] GetNose(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var eye = new Point[9];

            for (int i = 0; i < 9; i++)
            {
                eye[i] = points[i + 27];
            }

            return eye;
        }

        #endregion

        #region Symmetry and rotation angle

        /// <summary>
        /// Returns rotation angle from points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Angle</returns>
        public static float GetRotationAngle(this Point[] points)
        {
            var left = Points.GetMeanPoint(points.GetLeftEye());
            var right = Points.GetMeanPoint(points.GetRightEye());
            var point = left.GetSupportedPoint(right);
            var angle = left.GetAngle(right, point);

            return angle;
        }

        /// <summary>
        /// Returns symmetry coefficient of face.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Symmetry coefficient [0, 1]</returns>
        public static float GetSymmetryCoefficient(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

            var nose = points.GetNose();
            var leftEye = points.GetLeftEye();
            var rightEye = points.GetRightEye();

            var noseCenterPoint = nose[0];

            var leftEyeLeftPoint = leftEye[0];
            var leftEyeRightPoint = leftEye[3];

            var rightEyeLeftPoint = rightEye[0];
            var rightEyeRightPoint = rightEye[3];

            var mouthUpperSymmetry = GetSymmetry(noseCenterPoint, leftEyeLeftPoint, rightEyeRightPoint);
            var mouthLowerSymmetry = GetSymmetry(noseCenterPoint, leftEyeRightPoint, rightEyeLeftPoint);

            return (mouthUpperSymmetry + mouthLowerSymmetry) / 2.0f;
        }

        /// <summary>
        /// Returns symmetry.
        /// </summary>
        /// <param name="a">Center point</param>
        /// <param name="b">Value</param>
        /// <param name="c">Value</param>
        /// <returns>Value</returns>
        private static float GetSymmetry(Point a, Point b, Point c)
        {
            var distLeft = Abs(a, b);
            var distRight = Abs(a, c);

            return GetSymmetry(distLeft, distRight);
        }

        /// <summary>
        /// Returns symmetry.
        /// </summary>
        /// <param name="a">Value</param>
        /// <param name="b">Value</param>
        /// <returns>Value</returns>
        private static float GetSymmetry(float a, float b)
        {
            var v = a / b;
            return v > 1.0 ? 1.0f / v : v;
        }

        /// <summary>
        /// Returns distance for two points.
        /// </summary>
        /// <param name="a">Point</param>
        /// <param name="b">Point</param>
        /// <returns>Value</returns>
        private static float Abs(Point a, Point b)
        {
            var abX = a.X - b.X;
            var abY = a.Y - b.Y;
            return (float)Maths.Sqrt(abX * abX + abY * abY);
        }

        #endregion
    }
}
