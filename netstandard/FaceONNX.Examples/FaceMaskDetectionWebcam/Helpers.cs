using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Media.Imaging;
using UMapx.Video;
using UMapx.Video.DirectShow;

namespace FaceMaskDetectionWebcam
{
    public static class Helpers
    {
        public static IVideoSource GetDevice(int camindex, int resindex)
        {
            try
            {
                var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice); int i = 0;
                var videoDevice = new VideoCaptureDevice(videoDevices[camindex].MonikerString);
                var videoCapabilities = videoDevice.VideoCapabilities;
                videoDevice.VideoResolution = videoCapabilities[resindex];
                return videoDevice;
            }
            catch
            {
                return null;
            }
        }

        public static BitmapImage ToBitmapImage(Bitmap bitmap)
        {
            var bi = new BitmapImage();
            bi.BeginInit();
            var ms = new MemoryStream();
            bitmap.Save(ms, ImageFormat.Bmp);
            ms.Seek(0, SeekOrigin.Begin);
            bi.StreamSource = ms;
            bi.EndInit();
            return bi;
        }
    }
}
