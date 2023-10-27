using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Defines face detector interface.
    /// </summary>
    public interface IFaceDetector : IDisposable
    {
        #region Interface

        /// <summary>
        /// Gets or sets confidence threshold.
        /// </summary>
        float ConfidenceThreshold { get; set; }

        /// <summary>
        /// Gets or sets NonMaxSuppression threshold.
        /// </summary>
        float NmsThreshold { get; set; }

        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>FaceDetectionResult</returns>
        FaceDetectionResult[] Forward(Bitmap image);

        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <returns>FaceDetectionResult</returns>
        FaceDetectionResult[] Forward(float[][,] image);

        #endregion
    }
}
