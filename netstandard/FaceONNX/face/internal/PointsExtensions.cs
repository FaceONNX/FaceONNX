using System.Drawing;
using UMapx.Core;

namespace FaceONNX
{
    /// <summary>
    /// Using for operations with points.
    /// </summary>
    internal static class PointsExtensions
    {
        /// <summary>
        /// Returns distance for two points.
        /// </summary>
        /// <param name="a">Point</param>
        /// <param name="b">Point</param>
        /// <returns>Value</returns>
        public static float Abs(this Point a, Point b)
        {
            var abX = a.X - b.X;
            var abY = a.Y - b.Y;
            return (float)Maths.Sqrt(abX * abX + abY * abY);
        }

        /// <summary>
        /// Returns symmetry.
        /// </summary>
        /// <param name="a">Center point</param>
        /// <param name="b">Value</param>
        /// <param name="c">Value</param>
        /// <returns>Value</returns>
        public static float GetSymmetry(this Point a, Point b, Point c)
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
        public static float GetSymmetry(this float a, float b)
        {
            var v = a / b;
            return v > 1.0 ? 1.0f / v : v;
        }
    }
}
