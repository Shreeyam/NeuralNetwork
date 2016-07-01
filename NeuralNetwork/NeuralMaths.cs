using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public static class NeuralMaths
    {
        public static double Logistic(double x) => 1 / (1 + Math.Exp(-x));
        public static IEnumerable<double> Logistic(IEnumerable<double> x)
        {
            foreach (double y in x)
            {
                yield return 1 / (1 + Math.Exp(-y));
            }
        }
        public static double LogisticGradient(double x) => x * (1 - x);

        public static IEnumerable<double> LogisticGradient(IEnumerable<double> x)
        {
            foreach (double y in x)
            {
                yield return Logistic(y) * (1 - Logistic(y));
            }
        }


        public static double NextGaussian(this Random rand)
        {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            return (Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
        }

        public static double Dot(double[] d1, double[] d2) => d1.Zip(d2, (x, y) => x * y).Sum();

        public static IEnumerable<double> DotMatrixVector(double[][] m1, double[] m2)
        {
            for (int i = 0; i < m1.Length; i++)
            {
                double Sum = 0;
                for (int j = 0; j < m1.First().Length; j++)
                {
                    Sum += m1[i][j] * m2[j];
                }
                yield return Sum;
            }
        }
    }
}
