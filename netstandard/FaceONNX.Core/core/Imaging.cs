using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace FaceONNX.Core
{
    /// <summary>
    /// Using for imaging.
    /// </summary>
    public static class Imaging
    {
        #region BitmapData voids
        /// <summary>
        /// Blocks Bitmap in system memory.
        /// </summary>
        /// <param name="b">Bitmap</param>
        /// <returns>Bitmap data</returns>
        public static BitmapData Lock24bpp(Bitmap b)
        {
            return b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
        }
        /// <summary>
        /// Unblocks Bitmap in system memory.
        /// </summary>
        /// <param name="b">Bitmap</param>
        /// <param name="bmData">Bitmap data</param>
        public static void Unlock(Bitmap b, BitmapData bmData)
        {
            b.UnlockBits(bmData);
            return;
        }
        #endregion

        #region Transformations
        /// <summary>
        /// Returns new bitmap.
        /// </summary>
        /// <param name="size">Size</param>
        /// <param name="color">Color</param>
        /// <returns>Bitmap</returns>
        public static Bitmap CreateBitmap(this Size size, Color color)
        {
            var bitmap = new Bitmap(size.Width, size.Height);
            using var graphics = Graphics.FromImage(bitmap);
            graphics.Clear(color);
            return bitmap;
        }
        /// <summary>
        /// Merges two images.
        /// </summary>
        /// <param name="background">Background image</param>
        /// <param name="foreground">Foreground image</param>
        /// <param name="rectangle">Rectangle</param>
        public static void Merge(this Bitmap background, Bitmap foreground, Rectangle rectangle)
        {
            using var graphics = Graphics.FromImage(background);
            graphics.DrawImage(foreground, rectangle);
            return;
        }
        /// <summary>
        /// Returns cropped image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Crop(this Bitmap image, Rectangle rectangle)
        {
            // image params
            int width = image.Width;
            int height = image.Height;

            // check section params
            int x = Range(rectangle.X, 0, width);
            int y = Range(rectangle.Y, 0, height);
            int w = Range(rectangle.Width, 0, width - x);
            int h = Range(rectangle.Height, 0, height - y);

            // fixes rectangle section
            var rectangle_fixed = new Rectangle(x, y, w, h);

            // crop image to rectangle section
            var bitmap = new Bitmap(rectangle_fixed.Width, rectangle_fixed.Height);
            var section = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            using var g = Graphics.FromImage(bitmap);
            g.DrawImage(image, section, rectangle_fixed, GraphicsUnit.Pixel);

            return bitmap;
        }
        /// <summary>
        /// Fixes value in range.
        /// </summary>
        /// <param name="x">Value</param>
        /// <param name="min">Min</param>
        /// <param name="max">Max</param>
        /// <returns>Value</returns>
        private static int Range(int x, int min, int max)
        {
            if (x < min)
            {
                return min;
            }
            else if (x > max)
            {
                return max;
            }
            return x;
        }
        /// <summary>
        /// Returns resized image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="size">Size</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Resize(this Bitmap image, Size size)
        {
            return new Bitmap(image, size.Width, size.Height);
        }
        /// <summary>
        /// Returns resized image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="size">Size</param>
        /// <param name="color">Border color</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Resize(this Bitmap image, Size size, Color color)
        {
            // size
            int width = image.Width;
            int height = image.Height;
            int max = Math.Max(width, height);

            //  borders
            var rectangle = new Rectangle(
                (max - width) / 2,
                (max - height) / 2,
                width,
                height);

            // drawing
            Bitmap background = new Bitmap(max, max);

            using var g = Graphics.FromImage(background);
            g.Clear(color);
            g.DrawImage(image, rectangle);

            return Imaging.Resize(background, size);
        }
        /// <summary>
        /// Returns rotated image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <param name="angle">Angle</param>
        /// <param name="color">Background color</param>
        /// <returns>Bitmap</returns>
        public static Bitmap Rotate(this Bitmap image, float angle, Color color)
        {
            // create an empty Bitmap image
            Bitmap bmp = new Bitmap(image.Width, image.Height);

            // turn the Bitmap into a Graphics object
            using var g = Graphics.FromImage(bmp);
            g.Clear(color);

            // now we set the rotation point to the center of our image
            g.TranslateTransform((float)bmp.Width / 2, (float)bmp.Height / 2);

            // now rotate the image
            g.RotateTransform(angle);
            g.TranslateTransform(-(float)bmp.Width / 2, -(float)bmp.Height / 2);

            // set the InterpolationMode to HighQualityBicubic so to ensure a high
            // quality image once it is transformed to the specified size
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;

            // now draw our new image onto the graphics object
            g.DrawImage(image, new Point(0, 0));

            //return the image
            return bmp;
        }
        /// <summary>
        /// Returns flipped by X image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Bitmap</returns>
        public static Bitmap FlipX(this Bitmap image)
        {
            Bitmap bmp = new Bitmap(image);
            bmp.RotateFlip(RotateFlipType.RotateNoneFlipX);
            return bmp;
        }
        /// <summary>
        /// Returns flipped by Y image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Bitmap</returns>
        public static Bitmap FlipY(this Bitmap image)
        {
            Bitmap bmp = new Bitmap(image);
            bmp.RotateFlip(RotateFlipType.RotateNoneFlipY);
            return bmp;
        }
        /// <summary>
        /// Returns flipped by XY image.
        /// </summary>
        /// <param name="image">Bitmap</param>
        /// <returns>Bitmap</returns>
        public static Bitmap FlipXY(this Bitmap image)
        {
            Bitmap bmp = new Bitmap(image);
            bmp.RotateFlip(RotateFlipType.RotateNoneFlipXY);
            return bmp;
        }
        #endregion

        #region Rectangles
        /// <summary>
        /// Returns size area.
        /// </summary>
        /// <param name="size">Size</param>
        /// <returns>Area</returns>
        public static int Area(this Size size)
        {
            return size.Width * size.Height;
        }
        /// <summary>
        /// Returns rectangle area.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <returns>Area</returns>
        public static int Area(this Rectangle rectangle)
        {
            return rectangle.Width * rectangle.Height;
        }
        /// <summary>
        /// Returns the maximum rectangle.
        /// </summary>
        /// <param name="rectangles">Rectangles</param>
        /// <returns>Rectangle</returns>
        public static Rectangle Max(params Rectangle[] rectangles)
        {
            // params
            var length = rectangles.Length;
            var rectangle = Rectangle.Empty;
            var area = 0;
            var max = 0;

            // do job
            for (int i = 0; i < length; i++)
            {
                rectangle = rectangles[i];

                if (rectangle.IsEmpty)
                    continue;

                if (rectangle.Area() > area)
                {
                    max = i;
                }
            }

            // output
            return length > 0 ? rectangles[max] : rectangle;
        }
        /// <summary>
        /// Returns the minimum rectangle.
        /// </summary>
        /// <param name="rectangles">Rectangles</param>
        /// <returns>Rectangle</returns>
        public static Rectangle Min(params Rectangle[] rectangles)
        {
            // params
            var length = rectangles.Length;
            var rectangle = Rectangle.Empty;
            var area = int.MaxValue;
            var min = int.MaxValue;

            // do job
            for (int i = 0; i < length; i++)
            {
                rectangle = rectangles[i];

                if (rectangle.IsEmpty)
                    continue;

                if (rectangle.Area() < area)
                {
                    min = i;
                }
            }

            // output
            return length > 0 ? rectangles[min] : rectangle;
        }
        /// <summary>
        /// Returns rectangle scaled to box.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <returns>Rectangle</returns>
        public static Rectangle ToBox(this Rectangle rectangle)
        {
            var max = Math.Max(rectangle.Width, rectangle.Height);
            var dx = max - rectangle.Width;
            var dy = max - rectangle.Height;

            return new Rectangle()
            {
                X = rectangle.X - dx / 2,
                Y = rectangle.Y - dy / 2,
                Width = rectangle.Width + dx,
                Height = rectangle.Height + dy
            };
        }
        /// <summary>
        /// Returns rectangle scaled to box.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="scale">Factor</param>
        /// <returns>Rectangle</returns>
        public static Rectangle ToBox(this Rectangle rectangle, float scale)
        {
            float gainX = rectangle.Width * scale;
            float gainY = rectangle.Height * scale;

            return new Rectangle(
                (int)(rectangle.X - gainX / 2),
                (int)(rectangle.Y - gainY / 2),
                (int)(rectangle.Width + gainX),
                (int)(rectangle.Height + gainY)
                );
        }
        /// <summary>
        /// Returns rectangle scaled to box.
        /// </summary>
        /// <param name="rectangles">Rectangle</param>
        /// <returns>Rectangle</returns>
        public static Rectangle[] ToBox(params Rectangle[] rectangles)
        {
            int length = rectangles.Length;
            var newRectangles = new Rectangle[length];

            for (int i = 0; i < length; i++)
            {
                newRectangles[i] = Imaging.ToBox(rectangles[i]);
            }

            return newRectangles;
        }
        /// <summary>
        /// Returns rectangle scaled to box with image size.
        /// </summary>
        /// <param name="rectangles">Rectangles</param>
        /// <param name="factor">Factor</param>
        /// <returns>Rectangle</returns>
        public static Rectangle[] ToBox(float factor, params Rectangle[] rectangles)
        {
            int length = rectangles.Length;
            var newRectangles = new Rectangle[length];

            for (int i = 0; i < length; i++)
            {
                newRectangles[i] = Imaging.ToBox(rectangles[i], factor);
            }

            return newRectangles;
        }
        /// <summary>
        /// Implements IoU operator.
        /// </summary>
        /// <param name="a">First rectangle</param>
        /// <param name="b">Second rectangle</param>
        /// <returns>Value</returns>
        public static float IoU(this Rectangle a, Rectangle b)
        {
            var xA = Math.Max(a.Left, b.Left);
            var yA = Math.Max(a.Top, b.Top);
            var xB = Math.Min(a.Right, b.Right);
            var yB = Math.Min(a.Bottom, b.Bottom);

            var interArea = Math.Abs(Math.Max(xB - xA, 0) * Math.Max(yB - yA, 0));

            if (interArea == 0)
                return 0;

            var boxAArea = Math.Abs((a.Right - a.Left) * (float)(a.Bottom - a.Top));
            var boxBArea = Math.Abs((b.Right - b.Left) * (float)(b.Bottom - b.Top));

            return interArea / (float)(boxAArea + boxBArea - interArea);
        }
        #endregion
    }
}
