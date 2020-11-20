using FaceONNX;
using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace FaceLandmarksExtraction
{
    public partial class Form1 : Form
    {
        FaceDetectorLight faceDetectorLight;
        FaceLandmarksExtractor faceLandmarksExtractor;

        public Form1()
        {
            InitializeComponent();
            DragDrop += Form1_DragDrop;
            DragEnter += Form1_DragEnter;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            faceDetectorLight = new FaceDetectorLight(0.75f, 0.25f);
            faceLandmarksExtractor = new FaceLandmarksExtractor();
        }

        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
                e.Effect = DragDropEffects.All;
            else
                e.Effect = DragDropEffects.None;
            return;
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            var file = ((string[])e.Data.GetData(DataFormats.FileDrop, true))[0];
            Bitmap image = new Bitmap(file);

            var faces = faceDetectorLight.Forward(image);

            foreach (var face in faces)
            {
                var color = GetRandomColor();
                var depth = 4;

                var pen = new Pen(color, depth);
                var points = faceLandmarksExtractor.Forward(image, face).First();
                Imaging.Draw(image, pen, face);
                Imaging.Draw(image, pen, depth, points);
            }

            BackgroundImage = image;
        }

        private static Color GetRandomColor()
        {
            var random = new Random();
            return Color.FromArgb(255,
                    128 + random.Next(128),
                    128 + random.Next(128),
                    128 + random.Next(128));
        }
    }
}
