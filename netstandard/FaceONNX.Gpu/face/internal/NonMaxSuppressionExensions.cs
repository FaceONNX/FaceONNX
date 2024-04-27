using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using UMapx.Imaging;

namespace FaceONNX
{
    /// <summary>
    /// Using for NonMaxSuppression operations.
    /// </summary>
    internal static class NonMaxSuppressionExensions
    {
        /// <summary>
        /// Agnostic NMS filtration (without regard classes of recognized objects).
        /// </summary>
        /// <param name="results">Results</param>
        /// <param name="nmsThreshold">Threshold</param>
        /// <returns>Results</returns>
        public static List<float[]> AgnosticNMSFiltration(this List<float[]> results, float nmsThreshold)
        {
            var list = results.OrderByDescending(x => x[4]).ToList();
            var length = list.Count;

            for (int i = 0; i < length; i++)
            {
                var first = list[i];

                for (int j = i + 1; j < length; j++)
                {
                    var second = list[j];

                    var iou = Rectangles.IoU(
                        Rectangle.FromLTRB(
                        (int)first[0],
                        (int)first[1],
                        (int)first[2],
                        (int)first[3]),

                        Rectangle.FromLTRB(
                        (int)second[0],
                        (int)second[1],
                        (int)second[2],
                        (int)second[3]
                        ));

                    if (iou > nmsThreshold)
                    {
                        list.RemoveAt(j);
                        length = list.Count;
                        j--;
                    }
                }
            }
            return list;
        }
    }
}
