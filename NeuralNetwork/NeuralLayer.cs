using System;
using System.Linq;
using static NeuralNetwork.NeuralMaths;

namespace NeuralNetwork
{
    public class NeuralLayer
    {
        public Neuron[] neurons;
        public double[] Outputs { get; private set; }

        /// <summary>
        /// The number of neurons
        /// </summary>
        public int NeuronCount
        {
            get { return neurons.Length; }
        }

        public NeuralLayer(int count, int previousLayerCount, Random rand)
        {
            neurons = new Neuron[count];
            Outputs = new double[count];
            for (int i = 0; i < count; i++)
            {
                neurons[i] = new Neuron();
                neurons[i].Initialize(previousLayerCount, rand);
            }
        }

        /// <summary>
        /// Returns the outputs for this layer only
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] Assess(double[] inputs)
        {
            for (int i = 0; i < NeuronCount; i++)
            {
                Outputs[i] = neurons[i].Assess(inputs);
            }
            return Outputs;
        }

        /// <summary>
        /// Updates neural layer and returns next error derivatives
        /// </summary>
        /// <param name="desired"></param>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public virtual double[] Update(double[] desired, double[] inputs, double learnFactor)
        {
            double[] inputsSum = new double[inputs.Length];
            for (int i = 0; i < NeuronCount; i++)
            {
                // Absolute error
                double error = -neurons[i].CalculateError(desired[i]);

                // Required change in weights
                double derivative = LogisticGradient(error);

                for (int j = 0; j < neurons[i].Weights.Length; j++)
                {
                    neurons[i].Weights[j] += learnFactor * derivative * inputs[j];
                    inputsSum[j] += neurons[i].Weights[j];
                }
            }
            return inputsSum;
        }

        /// <summary>
        /// Updates a neural layer with provided errors (use this if you have your own error function)
        /// </summary>
        /// <param name="errors"></param>
        /// <param name="inputs"></param>
        /// <param name="learnFactor"></param>
        /// <returns></returns>
        public virtual double[] UpdateWithProvidedErrors(double[] errors, double[] inputs, double learnFactor)
        {
            double[] inputsSum = new double[inputs.Length];
            for (int i = 0; i < NeuronCount; i++)
            {
                // Absolute error
                double error = errors[i];

                // Required change in weights
                double derivative = LogisticGradient(error);

                for (int j = 0; j < neurons[i].Weights.Length; j++)
                {
                    neurons[i].Weights[j] += learnFactor * derivative * inputs[j];
                    inputsSum[j] += neurons[i].Weights[j];
                }
            }
            return inputsSum;
        }

        public virtual double[] CalculateDifference(double desired) => neurons.Select(x => desired - x.Output).ToArray();

        public virtual double[] CalculateErrors(double[] desired) => neurons.Select((x, i) => x.CalculateError(desired[i])).ToArray();
    }
}
