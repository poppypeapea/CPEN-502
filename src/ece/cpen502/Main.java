package ece.cpen502;

import ece.cpen502.NN.NeuralNet;

public class Main {
    public static void main(String[] args) {
        // Set up your network with the specified configuration
        NeuralNet neuralNet = new NeuralNet();

        int numTrials = 100; // Number of trials to perform

        // Binary representation trials
//        double binaryAvgEpochs = runTrials(neuralNet, numTrials, false);

        // Bipolar representation trials
        double bipolarAvgEpochs = runTrials(neuralNet, numTrials, true);

//        System.out.println("Average epochs for binary representation: " + binaryAvgEpochs);
        System.out.println("Average epochs for bipolar representation: " + bipolarAvgEpochs);
    }

    private static double runTrials(NeuralNet neuralNet, int numTrials, boolean bipolar) {
        double totalEpochs = 0.0;

        for (int trial = 1; trial <= numTrials; trial++) {
            // Initialize the weights randomly
            neuralNet.initializeWeights();

            // Define the XOR training data
            double[][] inputTrainingData;
            double[] targetOutputData;

            if (bipolar) {
                // Define the XOR training data with bipolar representation
                inputTrainingData = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
                targetOutputData = new double[]{-1, 1, 1, -1};
            } else {
                // Define the XOR training data with binary representation
                inputTrainingData = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
                targetOutputData = new double[]{0, 1, 1, 0};
            }

            // Train the neural network
            double totalError = 1.0;
            int epochs = 0;

            while (totalError > NeuralNet.MAX_ERROR && epochs < NeuralNet.MAX_EPOCH) {
                totalError = 0.0;

                for (int i = 0; i < inputTrainingData.length; i++) {
                    double[] input = inputTrainingData[i];
                    double targetOutput = targetOutputData[i];

                    double networkOutput = neuralNet.forwardPropagation(input);
                    neuralNet.backpropagation(input, targetOutput);

                    // Calculate the error (mean squared error)
                    totalError += Math.pow(targetOutput - networkOutput, 2) / 2;
                }

                epochs++;
            }

            totalEpochs += epochs;
        }

        return totalEpochs / numTrials;
    }
}
