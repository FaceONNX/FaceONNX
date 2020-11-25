using System.Drawing;

namespace FaceONNX
{
	/// <summary>
	/// Defines face detector interface.
	/// </summary>
    public interface IFaceDetector
    {
        #region Interface
        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Rectangles</returns>
        public Rectangle[] Forward(Bitmap image);
        #endregion
    }
}
