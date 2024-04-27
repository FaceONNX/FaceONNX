using System.Drawing;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Defines object detection result.
    /// </summary>
    public class FaceDetectionResult
    {
        /// <summary>
        /// Gets or sets label id.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets score.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Gets or sets rectangle.
        /// </summary>
        public Rectangle Rectangle { get; set; }

        /// <summary>
        /// Gets box.
        /// </summary>
        public Rectangle Box 
        {
            get
            {
                return Rectangle.ToBox();
            }
        }

        /// <summary>
        /// Gets or sets face 5 landmarks.
        /// </summary>
        public Face5Landmarks Points { get; set; }

        /// <summary>
        /// Empty object detection result.
        /// </summary>
        public static FaceDetectionResult Empty
        {
            get
            {
                return new FaceDetectionResult
                {
                    Rectangle = Rectangle.Empty,
                    Score = 0,
                    Id = -1,
                    Points = new Face5Landmarks(new Point[5])
                };
            }
        }
    }
}
