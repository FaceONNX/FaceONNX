using RealSenseFaceID.Core;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Threading;
using UMapx.Imaging;
using UMapx.Video;
using UMapx.Video.RealSense;
using UMapx.Visualization;

namespace RealSenseFaceID
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Fields

        private readonly RealSenseVideoSource _realSenseVideoSource;
        private static readonly object _locker = new();
        private readonly FaceID _faceVerification;
        private readonly Painter _painter;
        private Thread _thread;
        private Bitmap _frame;
        private ushort[,] _depth;

        #endregion

        #region Launcher

        /// <summary>
        /// Constructor.
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();
            Closing += MainWindow_Closing;

            _faceVerification = new FaceID(useEyeTracking: true, useCUDA: true);

            var files = Directory.GetFiles(@"..\..\..\images", "*.*", SearchOption.AllDirectories);

            foreach (var file in files)
            {
                using var bitmap = new Bitmap(file);
                var name = Path.GetFileNameWithoutExtension(file);
                _faceVerification.AddFace(bitmap, name);
            }

            _painter = new Painter();
            _realSenseVideoSource = new RealSenseVideoSource();
            _realSenseVideoSource.NewFrame += OnNewFrame;
            _realSenseVideoSource.NewDepth += OnNewDepth;
            _realSenseVideoSource.Start();
        }

        /// <summary>
        /// Windows closing.
        /// </summary>
        /// <param name="sender">Sender</param>
        /// <param name="e">Event args</param>
        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _realSenseVideoSource?.SignalToStop();
            _faceVerification?.Dispose();
            _painter?.Dispose();
            _realSenseVideoSource?.Dispose();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Get frame and dispose previous.
        /// </summary>
        private Bitmap Frame
        {
            get
            {
                if (_frame is null)
                    return null;

                Bitmap frame;

                lock (_locker)
                {
                    frame = (Bitmap)_frame.Clone();
                }

                return frame;
            }
            set
            {
                lock (_locker)
                {
                    if (_frame != null)
                    {
                        _frame.Dispose();
                        _frame = null;
                    }

                    _frame = value;
                }
            }
        }

        /// <summary>
        /// Gets depth and dispose previous.
        /// </summary>
        private ushort[,] Depth
        {
            get
            {
                return _depth;
            }
            set
            {
                _depth = value;
            }
        }

        /// <summary>
        /// Gets or sets verification result.
        /// </summary>
        private FaceIDResult VerificationResult { get; set; } = new FaceIDResult();

        #endregion

        #region Handling events

        /// <summary>
        /// Frame handling on event call.
        /// </summary>
        /// <param name="sender">sender</param>
        /// <param name="eventArgs">event arguments</param>
        private void OnNewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            Frame = (Bitmap)eventArgs.Frame.Clone();
            InvokeDrawing();

            _thread = new Thread(() => ProcessFrame(Frame, Depth))
            {
                IsBackground = true,
                Priority = ThreadPriority.Normal
            };
            _thread?.Start();
        }

        /// <summary>
        /// Depth handling on event call.
        /// </summary>
        /// <param name="sender">sender</param>
        /// <param name="eventArgs">event arguments</param>
        private void OnNewDepth(object sender, NewDepthEventArgs eventArgs)
        {
            Depth = (ushort[,])eventArgs.Depth.Clone();
            InvokeDrawing();
        }

        #endregion

        #region Private voids

        /// <summary>
        /// Process frame.
        /// </summary>
        /// <param name="frame">Bitmap</param>
        /// <param name="depth">Matrix</param>
        private void ProcessFrame(Bitmap frame, ushort[,] depth)
        {
            VerificationResult = _faceVerification.Forward(frame, depth);
            InvokeDrawing();
        }

        /// <summary>
        /// Invoke drawing method.
        /// </summary>
        private void InvokeDrawing()
        {
            // verification result
            var verificationResult = VerificationResult;
            var paintData = new PaintData
            {
                Rectangle = verificationResult.Rectangle,
                Points = verificationResult.Landmarks.All,
                Labels = new string[] { verificationResult.Label,
                        verificationResult.Live.ToString() }
            };

            // color drawing
            var printColor = Frame;

            if (printColor != null)
            {
                lock (_locker)
                {
                    using var graphics = Graphics.FromImage(printColor);
                    _painter.Draw(graphics, paintData);
                }

                var bitmapColor = printColor.ToBitmapSource();
                bitmapColor.Freeze();
                _ = Dispatcher.BeginInvoke(new ThreadStart(delegate { imgColor.Source = bitmapColor; }));
            }

            // depth drawing
            using var printDepth = Depth?.Equalize()?.FromDepth();

            if (printDepth != null)
            {
                lock (_locker)
                {
                    using var graphics = Graphics.FromImage(printDepth);
                    _painter.Draw(graphics, paintData);
                }

                var bitmapDepth = printDepth.ToBitmapSource();
                bitmapDepth.Freeze();
                _ = Dispatcher.BeginInvoke(new ThreadStart(delegate { imgDepth.Source = bitmapDepth; }));
            }
        }

        #endregion
    }
}
