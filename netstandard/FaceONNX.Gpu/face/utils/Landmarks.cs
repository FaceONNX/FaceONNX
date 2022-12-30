using System;
using System.Drawing;
using UMapx.Core;

namespace FaceONNX
{
    /// <summary>
    /// Using for face points operations.
    /// </summary>
    public static class Landmarks
    {
        #region Operators

        /// <summary>
        /// Returns processed points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="point">Point</param>
        /// <returns>Points</returns>
        public static Point[] Add(this Point[] points, Point point)
        {
            var count = points.Length;
            var output = new Point[count];

            for (int i = 0; i < count; i++)
            {
                output[i] = new Point
                {
                    X = points[i].X + point.X,
                    Y = points[i].Y + point.Y
                };
            }

            return output;
        }

        /// <summary>
        /// Returns processed points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="point">Point</param>
        /// <returns>Points</returns>
        public static Point[] Sub(this Point[] points, Point point)
        {
            var count = points.Length;
            var output = new Point[count];

            for (int i = 0; i < count; i++)
            {
                output[i] = new Point
                {
                    X = points[i].X - point.X,
                    Y = points[i].Y - point.Y
                };
            }

            return output;
        }

        #endregion

        #region Special operators

        /// <summary>
        /// Rotates points by angle.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="centerPoint">Center point</param>
        /// <param name="angle">Angle</param>
        /// <returns>Points</returns>
        public static Point[] Rotate(this Point[] points, Point centerPoint, float angle)
        {
            int length = points.Length;
            var output = new Point[length];

            for (int i = 0; i < length; i++)
            {
                output[i] = points[i].Rotate(centerPoint, angle);
            }

            return output;
        }

        /// <summary>
        /// Rotates point by angle.
        /// </summary>
        /// <param name="pointToRotate">The point to rotate.</param>
        /// <param name="centerPoint">The center point of rotation.</param>
        /// <param name="angleInDegrees">The rotation angle in degrees.</param>
        /// <returns>Rotated point</returns>
        public static Point Rotate(this Point pointToRotate, Point centerPoint, double angleInDegrees)
        {
            double angleInRadians = angleInDegrees * (Math.PI / 180);
            double cosTheta = Math.Cos(angleInRadians);
            double sinTheta = Math.Sin(angleInRadians);

            return new Point
            {
                X =
                    (int)
                    (cosTheta * (pointToRotate.X - centerPoint.X) -
                    sinTheta * (pointToRotate.Y - centerPoint.Y) + centerPoint.X),
                Y =
                    (int)
                    (sinTheta * (pointToRotate.X - centerPoint.X) +
                    cosTheta * (pointToRotate.Y - centerPoint.Y) + centerPoint.Y)
            };
        }

        /// <summary>
        /// Returns rectangle from points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Rectangle</returns>
        public static Rectangle GetRectangle(this Point[] points)
        {
            int length = points.Length;
            int xmin = int.MaxValue;
            int ymin = int.MaxValue;
            int xmax = int.MinValue;
            int ymax = int.MinValue;

            for (int i = 0; i < length; i++)
            {
                int x = points[i].X;
                int y = points[i].Y;

                if (x < xmin)
                    xmin = x;
                if (y < ymin)
                    ymin = y;
                if (x > xmax)
                    xmax = x;
                if (y > ymax)
                    ymax = y;
            }

            return new Rectangle(xmin, ymin, xmax - xmin, ymax - ymin);
        }

        /// <summary>
        /// Return angle of the three points.
        /// </summary>
        /// <param name="left">Left point</param>
        /// <param name="right">Right point</param>
        /// <param name="support">Supported point</param>
        /// <returns>Angle</returns>
        public static float GetAngle(this Point left, Point right, Point support)
        {
            double kk = left.Y > right.Y ? 1 : -1;

            double x1 = left.X - support.X;
            double y1 = left.Y - support.Y;

            double x2 = right.X - left.X;
            double y2 = right.Y - left.Y;

            double cos = (x1 * x2 + y1 * y2) / Math.Sqrt(x1 * x1 + y1 * y1) / Math.Sqrt(x2 * x2 + y2 * y2);
            return (float)(kk * (180.0 - Math.Acos(cos) * 57.3));
        }

        /// <summary>
        /// Returns supported point.
        /// </summary>
        /// <param name="left">Left point</param>
        /// <param name="right">Right point</param>
        /// <returns>Point</returns>
        public static Point GetSupportedPoint(this Point left, Point right)
        {
            return new Point(right.X, left.Y);
        }

        /// <summary>
        /// Returns mean point.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Point</returns>
        public static Point GetMeanPoint(params Point[] points)
        {
            var point = new Point(0, 0);
            var length = points.Length;

            for (int i = 0; i < length; i++)
            {
                point.X += points[i].X;
                point.Y += points[i].Y;
            }

            point.X /= length;
            point.Y /= length;

            return point;
        }
        #endregion

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
            var left = Landmarks.GetMeanPoint(points.GetLeftEye());
            var right = Landmarks.GetMeanPoint(points.GetRightEye());
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
