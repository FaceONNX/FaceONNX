using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Defines face 68 landmarks extractor interface.
    /// </summary>
    public interface IFace68LandmarksExtractor : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>FaceLandmarks</returns>
        Face68Landmarks Forward(Bitmap image);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="clamp">Clamp or not</param>
        /// <returns>FaceLandmarks</returns>
        Face68Landmarks Forward(Bitmap image, Rectangle rectangle, bool clamp = true);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>FaceLandmarks</returns>
        Face68Landmarks Forward(float[][,] image);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="clamp">Clamp or not</param>
        /// <returns>FaceLandmarks</returns>
        Face68Landmarks Forward(float[][,] image, Rectangle rectangle, bool clamp = true);

        #endregion
    }
}
