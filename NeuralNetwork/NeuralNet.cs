using System;
using System.Linq;

namespace NeuralNetwork
{
    /// <summary>
    /// Basic fixed-topology neural network code
    /// Copyright Shreeyam Kacker 2016
    /// All rights reserved
    /// </summary>
    public class NeuralNet
    {
        private NeuralLayer[] neuralLayers;
        Random rand = new Random();

        public int OutputCount { get; }
        public int LayerCount { get { return neuralLayers.Length; } }
        public int InputCount { get; }
        public double Fitness { get; set; }
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

        /// <summary>
        /// Forward propogate network and return output
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] Assess(double[] inputs)
        {
            //Fitness = AsessFitness.Invoke();
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

        /// <summary>
        /// Breed two neural networks
        /// </summary>
        /// <param name="n1"></param>
        /// <param name="n2"></param>
        /// <param name="mutationChance"></param>
        /// <returns></returns>
        public static NeuralNet Breed(NeuralNet n1, NeuralNet n2, double mutationChance)
        {
            return BreedNetworks(n1, n2, mutationChance);
        }

        /// <summary>
        /// Breed this network with another with a provided mutation chance
        /// </summary>
        /// <param name="n2"></param>
        /// <param name="mutationChance"></param>
        /// <returns></returns>
        public NeuralNet Breed(NeuralNet n2, double mutationChance)
        {
            return BreedNetworks(this, n2, mutationChance);
        }

        private static NeuralNet BreedNetworks(NeuralNet n1, NeuralNet n2, double mutationChance)
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


        /// <summary>
        /// Train the network on a data set for n iterations
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="iterations"></param>
        public void Classify(double[] inputs, double[] outputs, int iterations)
        {
            for (int e = 0; e < iterations; e++)
            {
                ClassifyOnce(inputs, outputs);
            }
        }

        /// <summary>
        /// Train the network on many data sets for n iterations
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="iterations"></param>
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
                        nextErrors = neuralLayers[l].Update(desiredOutputs, actualInputs, _learnFactor);
                    }
                    // Update for each neuron that isn't the output
                    else
                    {
                        nextErrors = neuralLayers[l].UpdateWithProvidedErrors(nextErrors, actualInputs, _learnFactor);
                    }
                }
            }
        }

    }
}
