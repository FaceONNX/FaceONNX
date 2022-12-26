using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Defines face classifier interface.
    /// </summary>
    public interface IFaceClassifier : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Array</returns>
        float[] Forward(Bitmap image);

        /// <summary>
        /// Returns face recognition results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>Array</returns>
        float[] Forward(float[][,] image);

        #endregion
    }
}
