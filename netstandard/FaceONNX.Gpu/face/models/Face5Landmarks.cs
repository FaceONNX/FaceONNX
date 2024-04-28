using System;
using System.Drawing;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Defines face 5 landmarks class.
    /// </summary>
    public class Face5Landmarks
    {
        #region Fields

        /// <summary>
        /// Face points.
        /// </summary>
        private readonly Point[] _points;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face 5 landmarks class.
        /// </summary>
        /// <param name="points">Points</param>
        /// <exception cref="ArgumentException">Exception of incorrect points array size</exception>
        public Face5Landmarks(Point[] points)
        {
            if (points.Length != 5)
                throw new ArgumentException("The number of face points must be 5.");

            _points = points;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns all face points.
        /// </summary>
        public Point[] All
        {
            get
            {
                return _points;
            }
        }

        /// <summary>
        /// Returns right eye points.
        /// </summary>
        public Point RightEye
        {
            get
            {
                return _points[1];
            }
        }

        /// <summary>
        /// Returns left eye points.
        /// </summary>
        public Point LeftEye
        {
            get
            {
                return _points[0];
            }
        }

        /// <summary>
        /// Returns mouth points.
        /// </summary>
        public Point[] Mouth
        {
            get
            {
                var tongue = new Point[2];

                for (int i = 0; i < 2; i++)
                {
                    tongue[i] = _points[i + 3];
                }

                return tongue;
            }
        }

        /// <summary>
        /// Returns nose points.
        /// </summary>
        public Point Nose
        {
            get
            {
                return _points[2];
            }
        }

        /// <summary>
        /// Returns rotation angle from points.
        /// </summary>
        public float RotationAngle
        {
            get
            {
                var left = Points.GetMeanPoint(LeftEye);
                var right = Points.GetMeanPoint(RightEye);
                var point = left.GetSupportedPoint(right);
                var angle = left.GetAngle(right, point);

                return angle;
            }
        }

        /// <summary>
        /// Returns symmetry coefficient of face.
        /// </summary>
        public float SymmetryCoefficient
        {
            get
            {
                return PointsExtensions.GetSymmetry(Nose, LeftEye, RightEye);
            }
        }

        #endregion
    }
}
