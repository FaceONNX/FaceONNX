using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Defines face landmarks extractor interface.
    /// </summary>
    public interface IFaceLandmarksExtractor : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>FaceLandmarks</returns>
        FaceLandmarks Forward(Bitmap image);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="clamp">Clamp or not</param>
        /// <returns>FaceLandmarks</returns>
        FaceLandmarks Forward(Bitmap image, Rectangle rectangle, bool clamp = true);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>FaceLandmarks</returns>
        FaceLandmarks Forward(float[][,] image);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="clamp">Clamp or not</param>
        /// <returns>FaceLandmarks</returns>
        FaceLandmarks Forward(float[][,] image, Rectangle rectangle, bool clamp = true);

        #endregion
    }
}
