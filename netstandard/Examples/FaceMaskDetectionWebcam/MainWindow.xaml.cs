using FaceONNX;
using System.Drawing;
using System.Threading;
using System.Windows;
using UMapx.Core;
using UMapx.Imaging;
using UMapx.Video;
using UMapx.Visualization;

namespace FaceMaskDetectionWebcam
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private static readonly object _locker = new object();
        private IVideoSource _videoSource;
        private FaceDetectorLight _faceDetector;
        private FaceLandmarksExtractor _faceLandmarksExtractor;
        private FaceMaskClassifier _faceMaskClassifier;
        private Painter _painter;
        private Bitmap _frame;
        private Rectangle[] _rectangles = { };
        private string[][] _protos = { };
        private Thread _thread;

        public MainWindow()
        {
            InitializeComponent();

            _faceDetector = new FaceDetectorLight();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _faceMaskClassifier = new FaceMaskClassifier();
            _painter = new Painter()
            {
                TextFont = new Font("Arial", 12),
                InsideBox = true
            };

            _videoSource = Helpers.GetDevice(0, 0);
            _videoSource.NewFrame += VideoSource_NewFrame;
            _videoSource.Start();
        }

        public Bitmap Frame
        {
            get
            {
                Bitmap frame;

                lock (_locker)
                    frame = (Bitmap)_frame.Clone();

                return frame;
            }
            set
            {
                lock (_locker)
                {
                    if (_frame is object)
                    {
                        _frame.Dispose();
                        _frame = null;
                    }

                    _frame = value;
                }
            }
        }

        public Rectangle[] Rectangles
        {
            get
            {
                lock (_locker)
                    return _rectangles;
            }
            set
            {
                lock (_locker)
                    _rectangles = value;
            }
        }

        public string[][] Protos
        {
            get
            {
                lock (_locker)
                    return _protos;
            }
            set
            {
                lock (_locker)
                    _protos = value;
            }
        }

        private void VideoSource_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            Frame = (Bitmap)eventArgs.Frame.Clone();
            InvokeDrawing();

            if (_thread == null)
            {
                _thread = new Thread(() => ProcessFrame(Frame))
                {
                    IsBackground = true,
                    Priority = ThreadPriority.Lowest
                };
                _thread.Start();
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (_videoSource is object)
            {
                _videoSource.NewFrame += VideoSource_NewFrame;
                _videoSource.SignalToStop();
                _videoSource = null;
                Thread.Sleep(1000);
            }

            _thread?.Join();

            _faceDetector?.Dispose();
            _faceLandmarksExtractor?.Dispose();
            _faceMaskClassifier?.Dispose();
        }

        private void InvokeDrawing()
        {
            var print = Frame;
            var length = Rectangles.Length;

            if (length == Protos.Length)
            {
                var paintData = new PaintData[length];

                for (int i = 0; i < length; i++)
                {
                    paintData[i] = new PaintData()
                    {
                        Rectangle = Rectangles[i],
                        Title = string.Empty,
                        Labels = Protos[i],
                    };
                }

                lock (_locker)
                {
                    using var graphics = Graphics.FromImage(print);
                    _painter.Draw(graphics, paintData);
                }
            }

            var bitmapImage = Helpers.ToBitmapImage(print);
            bitmapImage.Freeze();
            Dispatcher.BeginInvoke(new ThreadStart(delegate { CameraImage.Source = bitmapImage; }));
        }

        private void ProcessFrame(Bitmap imageFrame)
        {
            var rectangle = Imaging.Max(_faceDetector.Forward(imageFrame));

            if (!rectangle.IsEmpty)
            {
                using var cropped = BitmapTransform.Crop(imageFrame, rectangle);
                var points = _faceLandmarksExtractor.Forward(cropped);
                using var aligned = FaceLandmarksExtractor.Align(cropped, points);
                
                var rectangles = new Rectangle[] { rectangle };
                var predictions = _faceMaskClassifier.Forward(aligned);
                var maxPrediction = Matrice.Max(predictions, out int argmax);
                var label = FaceMaskClassifier.Labels[argmax];
                var proto = new string[][] { new[] { label } };

                Rectangles = rectangles;
                Protos = proto;

                InvokeDrawing();
            }

            imageFrame.Dispose();
            _thread = null;
        }

    }
}
