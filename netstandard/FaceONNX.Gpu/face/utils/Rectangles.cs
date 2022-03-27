using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Using for face boxes operations.
    /// </summary>
    public static class Rectangles
    {
        #region Operators

        /// <summary>
        /// Returns processed rectangle.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="point">Point</param>
        /// <returns>Rectangle</returns>
        public static Rectangle Add(this Rectangle rectangle, Point point)
        {
            return new Rectangle
            {
                X = rectangle.X + point.X,
                Y = rectangle.Y + point.Y,
                Width = rectangle.Width,
                Height = rectangle.Height
            };
        }

        /// <summary>
        /// Returns processed rectangle.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="point">Point</param>
        /// <returns>Rectangle</returns>
        public static Rectangle Sub(this Rectangle rectangle, Point point)
        {
            return new Rectangle
            {
                X = rectangle.X - point.X,
                Y = rectangle.Y - point.Y,
                Width = rectangle.Width,
                Height = rectangle.Height
            };
        }

        /// <summary>
        /// Returns processed rectangles.
        /// </summary>
        /// <param name="rectangles">Rectangles</param>
        /// <param name="point">Point</param>
        /// <returns>Rectangles</returns>
        public static Rectangle[] Add(this Rectangle[] rectangles, Point point)
        {
            var count = rectangles.Length;
            var output = new Rectangle[count];
            
            for (int i = 0; i < count; i++)
            {
                output[i] = rectangles[i].Add(point);
            }

            return output;
        }

        /// <summary>
        /// Returns processed rectangles.
        /// </summary>
        /// <param name="rectangles">Rectangles</param>
        /// <param name="point">Point</param>
        /// <returns>Rectangles</returns>
        public static Rectangle[] Sub(this Rectangle[] rectangles, Point point)
        {
            var count = rectangles.Length;
            var output = new Rectangle[count];

            for (int i = 0; i < count; i++)
            {
                output[i] = rectangles[i].Sub(point);
            }

            return output;
        }

        #endregion

        #region Special operators

        /// <summary>
        /// Returns point from rectangle.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <returns>Point</returns>
        public static Point GetPoint(this Rectangle rectangle)
        {
            return new Point
            {
                X = rectangle.X,
                Y = rectangle.Y
            };
        }

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
                newRectangles[i] = Rectangles.ToBox(rectangles[i]);
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
                newRectangles[i] = Rectangles.ToBox(rectangles[i], factor);
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

        /// <summary>
        /// Implements scale operator.
        /// </summary>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="kx">Factor for x axis</param>
        /// <param name="ky">Factor for y axis</param>
        /// <returns></returns>
        public static Rectangle Scale(this Rectangle rectangle, float kx = 0.0f, float ky = 0.0f)
        {
            var x = rectangle.X;
            var y = rectangle.Y;
            var w = rectangle.Width;
            var h = rectangle.Height;

            var dw = (int)(w * kx);
            var dh = (int)(h * ky);

            return new Rectangle
            {
                X = x - dw / 2,
                Y = y - dh / 2,
                Width = w + dw,
                Height = h + dh,
            };
        }

        #endregion
    }
}
