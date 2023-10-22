using FaceONNX;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using UMapx.Core;
using UMapx.Imaging;
using UMapx.Visualization;

namespace FaceRotationLandmarks
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

            Console.WriteLine("FaceONNX: Face rotation landmarks");
        }

        #endregion

        #region Form methods

        private void Form1_Load(object sender, EventArgs e)
        {
            var fileName = Path.Combine(Directory.GetCurrentDirectory(), "sources", "faces.jpg");
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

                var rectangles = _faceDetector.Forward(_bitmap);
                Console.WriteLine($"Detected {rectangles.Length} faces");

                for (int i = 0; i < rectangles.Length; i++)
                {
                    var points = _faceLandmarksExtractor.Forward(_bitmap, rectangles[i]);
                    var symmetry = FaceLandmarks.GetSymmetryCoefficient(points);
                    Console.WriteLine($"Face symmetry --> {symmetry}");

                    var paintData = new PaintData
                    {
                        Points = points.Add(new Point
                        {
                            X = rectangles[i].X,
                            Y = rectangles[i].Y
                        }),
                        Rectangle = rectangles[i],
                        Labels = new string[] { Math.Round(symmetry, 2).ToString() }
                    };

                    using var g = Graphics.FromImage(_bitmap);
                    _painter.Draw(g, paintData);
                }

                Console.WriteLine("Displaying results");
                this.BackgroundImage = _bitmap;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        #endregion
    }
}
