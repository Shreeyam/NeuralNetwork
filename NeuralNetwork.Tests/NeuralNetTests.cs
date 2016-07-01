using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork;
using System.Diagnostics;

namespace NeuralNetwork.Tests
{
    [TestClass]
    public class NeuralNetTests
    {
        [TestMethod]
        public void AssessTest()
        {
            // Arramge
            Neuron n = new Neuron();
            n.Weights = new double[] { 1, 2, 0.5, 0.25 };

            double[] inputs = new double[] { 2, 0.5, 3, 8 };

            // Act
            n.Assess(inputs);

            //Assert
            Assert.AreEqual(NeuralMaths.Logistic(6.5), n.Output);
        }

        [TestMethod]
        public void NewNeuralNetInitializesProperly()
        {
            // Arrange + Act
            NeuralNet neuralNet = new NeuralNet(6, 500, 10);

            // Assert
            Assert.AreEqual(6, neuralNet.LayerCount);
        }

        [TestMethod]
        public void NeuralNetReturnsOutputs()
        {
            // Arrange
            NeuralNet neuralNet = new NeuralNet(3, 10, 3);

            Random rand = new Random();
            double[] inputs = new double[10];
            for (int i = 0; i < 10; i++)
            {
                inputs[i] = rand.NextDouble();
            }

            // Act + Assert
            Assert.IsNotNull(neuralNet.Assess(inputs));
        }

        [TestMethod]
        public void NeuralNetworkImprovesInAccuracy()
        {
            // Arrange
            NeuralNet neuralNet = new NeuralNet(layers: 1, inputCount: 2, outputCount: 1);
            double[] inputs = new double[] { 1, -0.2 };

            double[] desired = new double[] { 0.05 };

            // Act
            double firstOutput = neuralNet.Assess(inputs).First();
            neuralNet.Classify(inputs, desired, 1000);
            double finalOutput = neuralNet.Assess(inputs).First();

            // Assert
            Assert.IsTrue(Math.Abs(desired.First() - firstOutput) > Math.Abs(desired.First() - finalOutput));
        }

        [TestMethod]
        public void NeuralNetworkImprovesInAccuracyForManyOutputs()
        {
            // Arrange
            NeuralNet neuralNet = new NeuralNet(2, 4, 2);
            double[] inputs = new double[] { 0, 0.5, 0.2, 0.8 };

            double[] desired = new double[] { 0.1 , 0.5};

            // Act
            double firstOutput = neuralNet.Assess(inputs).First();
            double firstOutputLast = neuralNet.Assess(inputs).Last();
            neuralNet.Classify(inputs, desired, 1000);
            double finalOutput = neuralNet.Assess(inputs).First();
            double finalOutputLast = neuralNet.Assess(inputs).Last();

            // Assert
            Assert.IsTrue(Math.Abs(desired.First() - firstOutput) > Math.Abs(desired.First() - finalOutput));
            Assert.IsTrue(Math.Abs(desired.Last() - firstOutputLast) > Math.Abs(desired.Last() - finalOutputLast));
        }

        [TestMethod]
        public void NeuralNetworkConvergesForMultipleConditions()
        {
            NeuralNet neuralNet = new NeuralNet(3, 4, 1);
            double[][] inputs =
            {
                new double[] { 0, 0, 0.5, 0 },
                new double[] { 1, 1, 1, 1}
            };

            double[][] desired =
            {
                new double[] { 1 },
                new double[] { 0.1 }
            };

            neuralNet.ClassifyMany(inputs, desired, 500000);

            for (int i = 0; i < inputs.Length; i++)
            {
                Assert.AreEqual(desired[i].First(), neuralNet.Assess(inputs[i]).First(), 0.05);
            }
        }

        [TestMethod]
        public void NeuralNetworksCanBreed()
        {
            // Arrange
            NeuralNet n1 = new NeuralNet(2, 100, 1);
            NeuralNet n2 = new NeuralNet(2, 100, 1);

            // Act + Assert
            Assert.IsNotNull(n1.Breed(n2, 0.05));
        }
    }
}
