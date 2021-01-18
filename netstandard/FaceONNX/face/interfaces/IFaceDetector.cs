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
        Rectangle[] Forward(Bitmap image);
        /// <summary>
        /// Disposes face detector.
        /// </summary>
        void Dispose();
        #endregion
    }
}
