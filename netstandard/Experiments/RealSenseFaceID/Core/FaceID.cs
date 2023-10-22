using FaceONNX;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Drawing;
using System.Linq;
using UMapx.Core;
using UMapx.Imaging;

namespace RealSenseFaceID.Core
{
    /// <summary>
    /// Face identification and verification class.
    /// </summary>
    public class FaceID : IDisposable
    {
        #region Fields

        private readonly IFaceDetector _faceDetector;
        private readonly IFaceLandmarksExtractor _faceLandmarksExtractor;
        private readonly IFaceClassifier _faceDepthClassifier;
        private readonly IFaceClassifier _eyeBlinkClassifier;
        private readonly IFaceClassifier _faceEmbedder;
        private readonly bool _useEyesTracking;
        private readonly Embeddings _embeddings;

        #endregion

        #region Consts

        private const float SimilarityThreshold = 0.51f;
        private const float DepthThreshold = 0.91f;
        private const float EyesThreshold = 0.051f;

        #endregion

        #region Constructor

        /// <summary>
        /// Face identification and verification class.
        /// </summary>
        /// <param name="useCUDA">Use CUDA or not</param>
        /// <param name="useEyeTracking">Use eye tracking or no</param>
        public FaceID(bool useEyeTracking = false, bool useCUDA = false)
        {
            var sessionOptions = useCUDA ? SessionOptions.MakeSessionOptionWithCudaProvider(0) : new SessionOptions();
            _faceDetector = new FaceDetector(sessionOptions);
            _faceLandmarksExtractor = new FaceLandmarksExtractor(sessionOptions);
            _faceDepthClassifier = new FaceDepthClassifier(sessionOptions);
            _eyeBlinkClassifier = new EyeBlinkClassifier(sessionOptions);
            _faceEmbedder = new FaceEmbedder(sessionOptions);
            _useEyesTracking = useEyeTracking;
            _embeddings = new Embeddings();
        }

        #endregion

        #region Methods

        /// <summary>
        /// Adds new face to embeddings database.
        /// </summary>
        /// <param name="frame">Bitmap</param>
        /// <param name="label">Label</param>
        /// <returns>Succes or not</returns>
        public bool AddFace(Bitmap frame, string label)
        {
            var rectangle = DetectFace(frame);

            if (rectangle.IsEmpty)
            {
                return false;
            }

            var result = ProcessFace(frame, rectangle, false);
            var vector = result.Item2;
            var output = _embeddings.FromSimilarity(vector);
            
            if (output.Item2 < SimilarityThreshold)
            {
                _embeddings.Add(vector, label);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Returns verification result.
        /// </summary>
        /// <param name="frame">Frame</param>
        /// <param name="depth">Depth</param>
        /// <returns></returns>
        public FaceIDResult Forward(Bitmap frame, ushort[,] depth)
        {
            // detection
            var rectangle = DetectFace(frame);

            if (rectangle.IsEmpty)
            {
                return new FaceIDResult();
            }

            // frame and depth processing
            var frame_result = ProcessFace(frame, rectangle, _useEyesTracking);
            var points = frame_result.Item1;
            var vector = frame_result.Item2;
            var frame_liveness = frame_result.Item3;

            // score label from database
            var output = _embeddings.FromSimilarity(vector);
            var similarity = output.Item2;
            var label = string.Empty;

            if (similarity > SimilarityThreshold)
            {
                label = output.Item1;
            }

            // depth processing
            var depth_liveness = ProcessFaceDepth(depth, rectangle, points);

            // output
            return new FaceIDResult
            {
                Label = label,
                Rectangle = rectangle,
                Landmarks = points.Add(rectangle.GetPoint()),
                Live = frame_liveness && depth_liveness
            };
        }

        #endregion

        #region Private methods

        /// <summary>
        /// Returns face detection results.
        /// </summary>
        /// <param name="frame">Bitmap</param>
        /// <returns>Output</returns>
        private Rectangle DetectFace(Bitmap frame)
        {
            var rectangles = _faceDetector.Forward(frame);
            var count = rectangles.Length;

            if (count > 0)
            {
                return Rectangles.Max(rectangles);
            }

            return Rectangle.Empty;
        }

        /// <summary>
        /// Returns face processing results.
        /// </summary>
        /// <param name="frame">Bitmap</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="useEyesTracking">Use eye tracking or not</param>
        /// <returns>Output</returns>
        private (Point[], float[], bool) ProcessFace(Bitmap frame, Rectangle rectangle, bool useEyesTracking)
        {
            // crop and align
            using var cropped = BitmapTransform.Crop(frame, rectangle);
            var points = _faceLandmarksExtractor.Forward(cropped);
            var angle = FaceLandmarks.GetRotationAngle(points);
            using var aligned = FaceLandmarksExtractor.Align(cropped, angle);

            // extract embeddings
            var vector = _faceEmbedder.Forward(aligned);
            var liveness = true;

            if (useEyesTracking)
            {
                // eye blink detection
                var eyes = EyeBlinkClassifier.GetEyesRectangles(points);
                using var left_eye = BitmapTransform.Crop(cropped, eyes.Item1);
                using var right_eye = BitmapTransform.Crop(cropped, eyes.Item2);
                var left_eye_value = _eyeBlinkClassifier.Forward(left_eye).First();
                var right_eye_value = _eyeBlinkClassifier.Forward(right_eye).First();

                liveness = left_eye_value > EyesThreshold
                    || right_eye_value > EyesThreshold;
            }

            return (points, vector, liveness);
        }

        /// <summary>
        /// Returns face depth processing results.
        /// </summary>
        /// <param name="depth">Depth</param>
        /// <param name="rectangle">Rectangle</param>
        /// <param name="points">Points</param>
        /// <returns>Output</returns>
        private bool ProcessFaceDepth(ushort[,] depth, Rectangle rectangle, Point[] points)
        {
            // crop and align
            using var depthCropped = DepthTransform.Crop(depth, rectangle).Equalize().FromDepth();
            var angle = FaceLandmarks.GetRotationAngle(points);
            using var depthCroppedAligned = FaceLandmarksExtractor.Align(depthCropped, angle);

            // classify
            var depth_output = _faceDepthClassifier.Forward(depthCroppedAligned);
            var depth_similarity = depth_output.Max(out int depth_index);

            var liveness = (depth_index == 1) &&
                (depth_similarity > DepthThreshold);

            return liveness;
        }

        #endregion

        #region IDisposable

        private bool _disposed;

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _faceDetector?.Dispose();
                    _faceLandmarksExtractor?.Dispose();
                    _faceDepthClassifier?.Dispose();
                    _eyeBlinkClassifier?.Dispose();
                    _faceEmbedder?.Dispose();
                }

                _disposed = true;
            }
        }

        ~FaceID()
        {
            Dispose(false);
        }

        #endregion
    }
}
