using System;
using System.Drawing;
using UMapx.Core;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Using for face processing.
    /// </summary>
    public static class FaceProcessingExtensions
    {
        #region Static methods

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="angle">Angle</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Align(this Bitmap image, float angle)
        {
            return image.Rotate(angle);
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="angle">Angle</param>
        /// <param name="clamp">Clamp crop or not</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Align(this Bitmap image, Rectangle rectangle, float angle, bool clamp = true)
        {
            var scaledRectangle = rectangle.Scale();
            using var cropped = image.Crop(scaledRectangle, clamp);
            using var aligned = Align(cropped, angle);
            var cropRectangle = rectangle.Sub(new Point
            {
                X = scaledRectangle.X,
                Y = scaledRectangle.Y
            });

            return aligned.Crop(cropRectangle, clamp);
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="angle">Angle</param>
        /// <returns>Image in BGR terms</returns>
        public static float[][,] Align(this float[][,] image, float angle)
        {
            var length = image.Length;

            if (length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var aligned = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                aligned[i] = image[i].Rotate(-angle);
            }

            return aligned;
        }

        /// <summary>
        /// Returns aligned face.
        /// </summary>
        /// <param name="image">Image in BGR terms</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="angle">Angle</param>
        /// <param name="clamp">Clamp crop or not</param>
        /// <returns>Image in BGR terms</returns>
        public static float[][,] Align(this float[][,] image, Rectangle rectangle, float angle, bool clamp = true)
        {
            var length = image.Length;

            if (length != 3)
                throw new ArgumentException("Image must be in BGR terms");

            var scaledRectangle = rectangle.Scale();
            var cropped = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                cropped[i] = image[i].Crop(
                    scaledRectangle.Y,
                    scaledRectangle.X,
                    scaledRectangle.Height,
                    scaledRectangle.Width, clamp);
            }

            var aligned = Align(cropped, angle);
            var cropRectangle = rectangle.Sub(new Point
            {
                X = scaledRectangle.X,
                Y = scaledRectangle.Y
            });

            var output = new float[length][,];

            for (int i = 0; i < length; i++)
            {
                output[i] = aligned[i].Crop(
                    cropRectangle.Y,
                    cropRectangle.X,
                    cropRectangle.Height,
                    cropRectangle.Width, clamp);
            }

            return output;
        }

        #endregion
    }
}
