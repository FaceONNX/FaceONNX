using FaceONNX;
using System.Drawing;

namespace RealSenseFaceID.Core
{
    /// <summary>
    /// Face identification and verification result.
    /// </summary>
    public class FaceIDResult
    {
        /// <summary>
        /// Face label.
        /// </summary>
        public string Label { get; set; } = string.Empty;

        /// <summary>
        /// Real face or not.
        /// </summary>
        public bool Live { get; set; } = false;

        /// <summary>
        /// Face rectangle.
        /// </summary>
        public Rectangle Rectangle { get; set; } = Rectangle.Empty;

        /// <summary>
        /// Face landmarks.
        /// </summary>
        public Face68Landmarks Landmarks { get; set; }
    }
}
