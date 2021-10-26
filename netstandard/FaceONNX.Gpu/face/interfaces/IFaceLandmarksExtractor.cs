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
        /// <param name="image">Image</param>
        /// <param name="rectangles">Rectangles</param>
        /// <returns>Point</returns>
        Point[][] Forward(Bitmap image, params Rectangle[] rectangles);

		/// <summary>
		/// Returns face recognition results.
		/// </summary>
		/// <param name="image">Bitmap</param>
		/// <returns>Point</returns>
		Point[] Forward(Bitmap image);

		#endregion
	}
}
