using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Using for face points operations.
    /// </summary>
    public static class Landmarks
    {
        #region Special operators
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
        /// Returns rectangle from face points.
        /// </summary>
        /// <param name="points">Points</param>
        /// <returns>Rectangle</returns>
        public static Rectangle GetRectangle(this Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("Face points are not correct.");

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
        /// Returns tongue points.
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
    }
}
