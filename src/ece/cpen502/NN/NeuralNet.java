package ece.cpen502.NN;

import ece.cpen502.Interface.NeuralNetInterface;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface {
    // Set up your network in a 2-input, 4-hidden and 1-output configuration
    private int numInputs = 2;
    private int numHidden = 4;
    private int numOutput = 1;

    private double [] hiddenInputVector;
    private double outPutVector;

    private double[][] hiddenWeights;
    private double[] outputWeights;

    // Set the learning rate to 0.2 with momentum at 0.0
    private double learningRate = 0.2;
    private double momentum = 0.0;

    // Define maximum epochs and maximum error
    // how many epochs does it take to reach a total error of less than 0.05?
    public static int MAX_EPOCH = 10000;
    public static double MAX_ERROR = 0.05;

    // if isBinary is True, it is binary; otherwise it is bipolar
    private boolean isBinary = false;

//     private boolean isBinary = true;

    // for momentum
    private double previousOutputWeight[];
    private double previousHiddenWeight[][];


    public NeuralNet() {
        hiddenInputVector = new double [numHidden];
        hiddenWeights = new double[numHidden][numInputs + 1]; // +1 to include bias weight
        outputWeights = new double[numHidden + 1]; // Output layer weights, +1 to include bias weight
        initializeWeights();
        previousOutputWeight = outputWeights;
        previousHiddenWeight = hiddenWeights;
    }

    private double calculateWeightedSum(double[] weights, double[] inputs) {
        // Calculate the weighted sum for computing neuron input
        double weightedSum = weights[0]; // Bias
        for (int i = 1; i < weights.length; i++) {
            weightedSum += weights[i] * inputs[i - 1];
        }
        return weightedSum;
    }

    private double calculateLayerOutput(double[] weights, double[] inputs) {
        // Calculate neuron output using Sigmoid activation function
        double weightedSum = calculateWeightedSum(weights, inputs);
        if (isBinary) {
            return sigmoid(weightedSum);
        } else { // bipolar
            return customSigmoid(weightedSum);
        }
    }

    /**
     * Perform a forward propagation step
     * @param inputVector Input vector
     * @return Neural network output
     */
    public double forwardPropagation(double[] inputVector) {
        if (outputWeights.length != hiddenInputVector.length + 1) throw new ArrayIndexOutOfBoundsException();
        else {
            // Input layer to hidden layer
            double[] hiddenLayerOutput = new double[hiddenWeights.length];
            for (int i = 0; i < this.hiddenWeights.length; i++) {
                double weightedSum = calculateWeightedSum(hiddenWeights[i], inputVector);
                if (isBinary) {
                    hiddenLayerOutput[i] = sigmoid(weightedSum);
                }
                else {
                    hiddenLayerOutput[i] = customSigmoid(weightedSum);
                }
            } hiddenInputVector = hiddenLayerOutput;

            // Hidden layer to output layer
            double networkOutput = calculateLayerOutput(outputWeights, hiddenLayerOutput);
            return networkOutput;
        }
    }

    public double sigmoid_derivative(double networkOutput) {
        // Calculate the derivative of the Sigmoid function
        if (isBinary){
            return networkOutput * (1 - networkOutput);
        }
        else{
            return (1 + networkOutput) * (1 - networkOutput);
        }
    }

    /**
     * Perform backpropagation
     * @param inputVector Input vector
     * @param targetOutput Target output
     */
    public void backpropagation(double[] inputVector, double targetOutput) {
        if (outputWeights.length != hiddenInputVector.length + 1) {
            throw new ArrayIndexOutOfBoundsException();
        }

        else {
            double networkOutput = forwardPropagation(inputVector);

            // Calculate the error (mean squared error)
            double error = targetOutput - networkOutput;

            // Calculate the gradient of the output layer
            double deltaOutput = error * sigmoid_derivative(networkOutput);

            /*
            // Update the weights and biases of the output layer
            outputWeights[0] += deltaOutput * learningRate; // Update bias
            for (int i = 1; i < this.outputWeights.length; i++) {
                outputWeights[i] += deltaOutput * learningRate * hiddenInputVector[i - 1];
            }

             */


            // momentum = 0.9
            // Weight updates for the output layer
            double weightChange[]  = new double [outputWeights.length]; // 初始化
            double newOutputWeight[] = new double[outputWeights.length];

            newOutputWeight[0] =  outputWeights[0] + momentum * (outputWeights[0] - previousOutputWeight[0]) +
                    deltaOutput * learningRate * 1;  // Update bias

            for (int i = 1; i < this.outputWeights.length; i++){
                weightChange[i] = outputWeights[i] - previousOutputWeight[i]; // current set of weights - previous set of weights
                newOutputWeight[i] = outputWeights[i] + momentum * weightChange[i] +
                        deltaOutput * learningRate * hiddenInputVector[i - 1];
            }

            // hidden momentum = 0.9
            double newHiddenWeight[][] = new double [hiddenWeights.length][numInputs + 1];

            double[] deltaHidden = new double[hiddenWeights.length];

            for (int i = 0; i < this.hiddenWeights.length; i++) {
                deltaHidden[i] = deltaOutput * outputWeights[i + 1] * sigmoid_derivative(hiddenInputVector[i]);
            }

            // Update the weights and biases of the hidden layer
            for (int i = 0; i < this.hiddenWeights.length; i++) {
                newHiddenWeight[i][0] = momentum * (hiddenWeights[i][0] - previousHiddenWeight[i][0]) +
                        deltaHidden[i] * learningRate; // Update bias
                for (int j = 1; j < this.hiddenWeights[i].length; j++) {
                    newHiddenWeight[i][j] = hiddenWeights[i][j] + momentum * (hiddenWeights[i][j] - previousHiddenWeight[i][j]) +
                            deltaHidden[i] * learningRate * inputVector[j - 1];
                }
            }

            // assignment
            previousOutputWeight = outputWeights;  //现在的给旧的
            outputWeights = newOutputWeight; //新的给现在

            previousHiddenWeight = hiddenWeights;
            hiddenWeights = newHiddenWeight;

            /*
            // Calculate the errors and gradients for the hidden layer
            double[] deltaHidden = new double[hiddenWeights.length];
            for (int i = 0; i < this.hiddenWeights.length; i++) {
                deltaHidden[i] = deltaOutput * outputWeights[i + 1] * sigmoid_derivative(hiddenInputVector[i]);
            }

            // Update the weights and biases of the hidden layer
            for (int i = 0; i < this.hiddenWeights.length; i++) {
                hiddenWeights[i][0] += deltaHidden[i] * learningRate; // Update bias
                for (int j = 1; j < this.hiddenWeights[i].length; j++) {
                    hiddenWeights[i][j] += deltaHidden[i] * learningRate * inputVector[j - 1];
                }
            }

             */
        }
    }

    @Override
    public double outputFor(double[] X) {
        // This method can be used to obtain the network's output but is not utilized in this example
        return 0;
    }

    @Override
    public double train(double[] X, double argValue) {
        // This method is also not utilized in this example
        return 0;
    }

    @Override
    public void save(File argFile) {
        // If you need to save the model, you can add code here
    }

    @Override
    public void load(String argFileName) throws IOException {
        // If you need to load a model, you can add code here
    }

    @Override
    public double sigmoid(double x) {
        // Sigmoid activation function
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double customSigmoid(double x) {
        return (double) (2) / (1 + Math.exp(-x)) + (-1);
    }

    /**
     * Initialize weights to random values in the range -0.5 to +0.5
     */
    @Override
    public void initializeWeights(){
    // Initialize weights with random values
        Random rand = new Random();

        // Initialize output layer weights
        for (int i = 0; i < outputWeights.length; i++) {
            outputWeights[i] = rand.nextDouble() - 0.5;
        }

        // Initialize hidden layer weights
        for (int i = 0; i < hiddenWeights.length; i++) {
            for (int j = 0; j < hiddenWeights[i].length; j++) {
                hiddenWeights[i][j] = rand.nextDouble() - 0.5;
            }
        }
    }

    @Override
    public void zeroWeights() {
        // If you need to reset weights to zero, you can add code here
    }
}
