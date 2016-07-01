using System;
using System.Linq;
using static NeuralNetwork.NeuralMaths;

namespace NeuralNetwork
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Output { get; set; }

        public double Assess(double[] inputs)
        {
            Output = Logistic(Weights.Zip(inputs, (x, y) => x * y).Sum());
            return Output;
        }

        public void Initialize(int previousLayerCount, Random rand)
        {
            Weights = new double[previousLayerCount];
            for (int i = 0; i < previousLayerCount; i++)
            {
                Weights[i] = rand.NextGaussian();
            }
        }

        public double CalculateError(double desired) => Output - desired;
    }
}
