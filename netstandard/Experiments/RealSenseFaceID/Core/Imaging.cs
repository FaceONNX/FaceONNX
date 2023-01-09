using System.Drawing;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using UMapx.Imaging;

namespace RealSenseFaceID.Core
{
    public static class Imaging
    {
        /// <summary>
        /// Converts Bitmap to BitmapSource.
        /// </summary>
        /// <param name="bitmap">Bitmap</param>
        /// <returns>BitmapSource</returns>
        public static BitmapSource ToBitmapSource(this Bitmap bitmap)
        {
            var bitmapData = BitmapFormat.Lock32bpp(bitmap);
            var bitmapSource = BitmapSource.Create(
                bitmapData.Width, bitmapData.Height,
                bitmap.HorizontalResolution, bitmap.VerticalResolution,
                PixelFormats.Bgra32, null,
                bitmapData.Scan0, bitmapData.Stride * bitmapData.Height, bitmapData.Stride);

            bitmap.Unlock(bitmapData);
            bitmapSource.Freeze();

            return bitmapSource;
        }
    }
}
