using FaceONNX;
using UMapx.Imaging;
using UMapx.Visualization;

namespace FaceAlignment
{
    public partial class Form1 : Form
    {
        #region Constructor

        public Form1()
        {
            InitializeComponent();

            this.AllowDrop = true;
            this.DragDrop += Form1_DragDrop;
            this.DragEnter += Form1_DragEnter;
            this.BackgroundImageLayout = ImageLayout.Zoom;

            _faceDetector = new FaceDetector();
            _faceLandmarksExtractor = new FaceLandmarksExtractor();
            _painter = new Painter
            {
                PointPen = new Pen(Color.Yellow, 6)
            };

            Console.WriteLine("FaceONNX: Face alignment");
        }

        #endregion

        #region Form methods

        private void Form1_Load(object sender, EventArgs e)
        {
            var fileName = Path.Combine(Directory.GetCurrentDirectory(), "sources", "jake.jpg");
            TryOpenImage(fileName);
        }

        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            var files = (string[])(e.Data.GetData(DataFormats.FileDrop, true));
            TryOpenImage(files.FirstOrDefault());
        }

        #endregion

        #region Private methods

        private void TryOpenImage(string fileName)
        {
            try
            {
                Console.WriteLine($"Image: {fileName}");
                _bitmap?.Dispose();
                _bitmap = new Bitmap(fileName);

                var faceDetectionResults = _faceDetector.Forward(_bitmap);
                Console.WriteLine($"Detected {faceDetectionResults.Length} faces");

                if (faceDetectionResults.Length <= 0)
                {
                    return;
                }

                var rectangle = Rectangles.Max(faceDetectionResults.Select(x => x.Box).ToArray());

                // naive alignment
                using var cropped = BitmapTransform.Crop(_bitmap, rectangle);
                var points1 = _faceLandmarksExtractor.Forward(cropped);
                var angle1 = points1.GetRotationAngle();

                _naiveAligned?.Dispose();
                _naiveAligned = FaceLandmarksExtractor.Align(cropped, angle1);

                // strong alignment
                var points2 = _faceLandmarksExtractor.Forward(_bitmap, rectangle);
                var angle2 = points2.GetRotationAngle();
                _strongAligned?.Dispose();
                _strongAligned = FaceLandmarksExtractor.Align(_bitmap, rectangle, angle2);

                // display results
                var paintData = new PaintData
                {
                    Points = points2.Add(new Point
                    {
                        X = rectangle.X,
                        Y = rectangle.Y
                    }),
                    Rectangle = rectangle
                };

                using var g = Graphics.FromImage(_bitmap);
                _painter.Draw(g, paintData);

                Console.WriteLine("Displaying results");
                this.pictureBox1.Image = _bitmap;
                this.pictureBox2.Image = _naiveAligned;
                this.pictureBox3.Image = _strongAligned;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        #endregion
    }
}