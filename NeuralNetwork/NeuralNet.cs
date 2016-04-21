using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeuralNetwork.NeuralMaths;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        private NeuralLayer[] neuralLayers;
        Random rand = new Random();

        public int OutputCount { get; }
        public int LayerCount { get { return neuralLayers.Length; } }
        public int InputCount { get; }
        public double Fitness { get; private set; }
        delegate double AsessFitness();
        private readonly double _learnFactor;

        public NeuralNet(int layers, int inputCount, int outputCount, double learnFactor = 1)
        {
            OutputCount = outputCount;
            InputCount = inputCount;
            neuralLayers = new NeuralLayer[layers];

            _learnFactor = learnFactor;
            double difference = inputCount - outputCount;

            //TODO: New layers here
            var _layerNodeNumbers = new int[layers];
            if (layers > 1)
            {
                for (int i = 0; i < layers; i++)
                {
                    _layerNodeNumbers[i] = Convert.ToInt32(inputCount - (difference * ((double)(i) / (layers - 1))));
                    neuralLayers[i] = new NeuralLayer(_layerNodeNumbers[i], (i == 0) ? inputCount : _layerNodeNumbers[i - 1], rand);
                }
            }
            else
            {
                _layerNodeNumbers[0] = outputCount;
                neuralLayers[0] = new NeuralLayer(_layerNodeNumbers[0], inputCount, rand);
            }
        }

        //public NeuralNet(int[] LayerNodeNumbers)
        //{
        //    for (int i = 1; i < LayerNodeNumbers.Count() - 1; i++)
        //    {
        //        neuralLayers[i] = new NeuralLayer(LayerNodeNumbers[i], ());
        //    }
        //}

        public double[] Assess(double[] inputs)
        {
            Fitness = AsessFitness.Invoke();
            double[] prevInputs;
            for (int i = 0; i < LayerCount; i++)
            {
                if (i == 0)
                {
                    prevInputs = neuralLayers[i].Assess(inputs);
                }
                else
                {
                    neuralLayers[i].Assess(neuralLayers[i - 1].Outputs);
                }
            }
            return neuralLayers.Last().Outputs;
        }

        public static NeuralNet Breed(NeuralNet n1, NeuralNet n2, double mutationChance)
        {
            Random rand = new Random();
            NeuralNet n = new NeuralNet(n1.LayerCount, n1.InputCount, n1.OutputCount);
            for (int i = 0; i < n1.LayerCount; i++)
            {
                for (int j = 0; j < n1.neuralLayers[i].NeuronCount; j++)
                {
                    for (int k = 0; k < n1.neuralLayers[i].neurons.First().Weights.Length; k++)
                    {
                        if (rand.NextDouble() < mutationChance)
                        {
                            n.neuralLayers[i].neurons[j].Weights[k] = rand.NextGaussian();
                        }
                        else
                        {
                            n.neuralLayers[i].neurons[j].Weights[k] = (n1.neuralLayers[i].neurons[j].Weights[k] + n2.neuralLayers[i].neurons[j].Weights[k]) / 2;
                        }
                    }
                }
            }

            return n;
        }

        public NeuralNet Breed(NeuralNet n2, double mutationChance)
        {
            Random rand = new Random();
            NeuralNet n = new NeuralNet(this.LayerCount, this.InputCount, this.OutputCount);
            for (int i = 0; i < this.LayerCount; i++)
            {
                for (int j = 0; j < this.neuralLayers[i].NeuronCount; j++)
                {
                    for (int k = 0; k < this.neuralLayers[i].neurons.First().Weights.Length; k++)
                    {
                        if (rand.NextDouble() < mutationChance)
                        {
                            n.neuralLayers[i].neurons[j].Weights[k] = rand.NextGaussian();
                        }
                        else
                        {
                            n.neuralLayers[i].neurons[j].Weights[k] = (this.neuralLayers[i].neurons[j].Weights[k] + n2.neuralLayers[i].neurons[j].Weights[k]) / 2;
                        }
                    }
                }
            }

            return n;
        }

        public void Classify(double[] inputs, double[] outputs, int iterations)
        {
            for (int e = 0; e < iterations; e++)
            {
                ClassifyOnce(inputs, outputs);
            }
        }

        public void ClassifyMany(double[][] inputs, double[][] outputs, int iterations)
        {
            for (int e = 0; e < iterations; e++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    ClassifyOnce(inputs[i], outputs[i]);
                }
            }
        }

        private void ClassifyOnce(double[] inputs, double[] outputs)
        {
            // Forward Propagation
            Assess(inputs);

            double[] nextDesiredOutputs = new double[0];
            for (int l = LayerCount - 1; l >= 0; l--)
            {
                double[] desiredOutputs = (l == LayerCount - 1) ? outputs : nextDesiredOutputs;
                double[] actualInputs = (l == 0) ? inputs : neuralLayers[l - 1].Outputs;

                double[] nextErrors = new double[actualInputs.Length];
                // For each output
                for (int o = 0; o < desiredOutputs.Length; o++)
                {
                    // Update for each output
                    if (l == LayerCount - 1)
                    {
                        nextErrors = neuralLayers[l].Update(desiredOutputs, actualInputs);
                    }
                    // Update for each neuron that isn't the output
                    else
                    {
                        nextErrors = neuralLayers[l].UpdateWithProvidedErrors(nextErrors, actualInputs);
                    }
                }
            }
        }

    }

    public class NeuralLayer
    {
        public Neuron[] neurons;
        public double[] Outputs { get; private set; }

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

        public double[] Assess(double[] inputs)
        {
            for (int i = 0; i < NeuronCount; i++)
            {
                Outputs[i] = neurons[i].Assess(inputs);
            }
            return Outputs;
        }

        //public void Update(double[] delta)
        //{
        //    for (int i = 0; i < NeuronCount; i++)
        //    {
        //        for (int j = 0; j < neurons[i].Weights.Length; j++)
        //        {
        //            neurons[i].Weights[j] += delta[0];
        //        }
        //    }
        //}

        /// <summary>
        /// Updates neural layer and returns next error derivatives
        /// </summary>
        /// <param name="desired"></param>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] Update(double[] desired, double[] inputs)
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
                    neurons[i].Weights[j] += derivative * inputs[j];
                    inputsSum[j] += neurons[i].Weights[j];
                }
            }
            return inputsSum;
        }

        public double[] UpdateWithProvidedErrors(double[] errors, double[] inputs)
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
                    neurons[i].Weights[j] += derivative * inputs[j];
                    inputsSum[j] += neurons[i].Weights[j];
                }
            }
            return inputsSum;
        }

        public double[] CalculateDifference(double desired) => neurons.Select(x => desired - x.Output).ToArray();

        public double[] CalculateErrors(double[] desired) => neurons.Select((x, i) => x.CalculateError(desired[i])).ToArray();
    }

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
