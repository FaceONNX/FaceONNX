using System;
using System.Drawing;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Defines face 68 landmarks class.
    /// </summary>
    public class Face68Landmarks
    {
        #region Fields

        /// <summary>
        /// Face points.
        /// </summary>
        private readonly Point[] _points;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face 68 landmarks class.
        /// </summary>
        /// <param name="points">Points</param>
        /// <exception cref="ArgumentException">Exception of incorrect points array size</exception>
        public Face68Landmarks(Point[] points)
        {
            if (points.Length != 68)
                throw new ArgumentException("The number of face points must be 68.");

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
        public Point[] RightEye
        {
            get
            {
                var eye = new Point[6];

                for (int i = 0; i < 6; i++)
                {
                    eye[i] = _points[i + 42];
                }

                return eye;
            }
        }

        /// <summary>
        /// Returns left eye points.
        /// </summary>
        public Point[] LeftEye
        {
            get
            {
                var eye = new Point[6];

                for (int i = 0; i < 6; i++)
                {
                    eye[i] = _points[i + 36];
                }

                return eye;
            }
        }

        /// <summary>
        /// Returns mouth points.
        /// </summary>
        public Point[] Mouth
        {
            get
            {
                var tongue = new Point[17];

                for (int i = 0; i < 17; i++)
                {
                    tongue[i] = _points[i + 48];
                }

                return tongue;
            }
        }

        /// <summary>
        /// Returns face points.
        /// </summary>
        public Point[] Face
        {
            get
            {
                var tongue = new Point[17];

                for (int i = 0; i < 17; i++)
                {
                    tongue[i] = _points[i];
                }

                return tongue;
            }
        }

        /// <summary>
        /// Returns left brow points.
        /// </summary>
        public Point[] LeftBrow
        {
            get
            {
                var eye = new Point[5];

                for (int i = 0; i< 5; i++)
                {
                    eye[i] = _points[i + 17];
                }

                return eye;
            }
        }

        /// <summary>
        /// Returns right brow points.
        /// </summary>
        public Point[] RightBrow
        {
            get
            {
                var eye = new Point[5];

                for (int i = 0; i < 5; i++)
                {
                    eye[i] = _points[i + 22];
                }

                return eye;
            }
        }

        /// <summary>
        /// Returns nose points.
        /// </summary>
        public Point[] Nose
        {
            get
            {
                var eye = new Point[9];

                for (int i = 0; i < 9; i++)
                {
                    eye[i] = _points[i + 27];
                }

                return eye;
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
                var nose = Nose;
                var leftEye = LeftEye;
                var rightEye = RightEye;

                var noseCenterPoint = nose[0];

                var leftEyeLeftPoint = leftEye[0];
                var leftEyeRightPoint = leftEye[3];

                var rightEyeLeftPoint = rightEye[0];
                var rightEyeRightPoint = rightEye[3];

                var mouthUpperSymmetry = PointsExtensions.GetSymmetry(noseCenterPoint, leftEyeLeftPoint, rightEyeRightPoint);
                var mouthLowerSymmetry = PointsExtensions.GetSymmetry(noseCenterPoint, leftEyeRightPoint, rightEyeLeftPoint);

                return (mouthUpperSymmetry + mouthLowerSymmetry) / 2.0f;
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Returns left eye rectangle from face landmarks.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="factor_x">Scale factor for OX</param>
        /// <param name="factor_y">Scale factor for OY</param>
        /// <returns>Rectangle</returns>
        public static Rectangle GetLeftEyeRectangle(Face68Landmarks points, float factor_x = 0.0f, float factor_y = 0.5f)
        {
            return points.LeftEye
                .GetRectangle()
                .Scale(factor_x, factor_y);
        }

        /// <summary>
        /// Returns right eye rectangle from face landmarks.
        /// </summary>
        /// <param name="points">Points</param>
        /// <param name="factor_x">Scale factor for OX</param>
        /// <param name="factor_y">Scale factor for OY</param>
        /// <returns>Rectangle</returns>
        public static Rectangle GetRightEyeRectangle(Face68Landmarks points, float factor_x = 0.0f, float factor_y = 0.5f)
        {
            return points.RightEye
                .GetRectangle()
                .Scale(factor_x, factor_y);
        }

        #endregion
    }
}
