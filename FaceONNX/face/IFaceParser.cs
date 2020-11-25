using System.Drawing;

namespace FaceONNX
{
    /// <summary>
	/// Defines face segmentation parser.
	/// </summary>
    public interface IFaceParser
    {
        #region Interface
        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image</param>
        /// <param name="rectangles">Rectangles</param>
        /// <returns>Array</returns>
        public float[][][,] Forward(Bitmap image, params Rectangle[] rectangles);
        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Array</returns>
        public unsafe float[][,] Forward(Bitmap image);
        #endregion
    }
}
