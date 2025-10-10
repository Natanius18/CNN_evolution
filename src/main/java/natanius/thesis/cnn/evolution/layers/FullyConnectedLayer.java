package natanius.thesis.cnn.evolution.layers;

import static natanius.thesis.cnn.evolution.data.Constants.OUTPUT_CLASSES;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.io.Serializable;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.activation.LeakyReLU;
import natanius.thesis.cnn.evolution.activation.ReLU;
import natanius.thesis.cnn.evolution.activation.Sigmoid;

public class FullyConnectedLayer extends Layer implements Serializable {

    private final Activation activation;

    private final double[][] weights;
    private final double[] biases;

    private final int inLength;
    private final double learningRate;

    private double[] lastZ;
    private double[] lastX;


    public FullyConnectedLayer(Activation activation, int inLength, double learningRate) {
        this.activation = activation;
        this.inLength = inLength;
        this.learningRate = learningRate;

        weights = new double[inLength][OUTPUT_CLASSES];
        if (activation instanceof ReLU || activation instanceof LeakyReLU) {
            initWeightsHe();
        } else if (activation instanceof Sigmoid) {
            initWeightsXavier();
        }

        biases = new double[OUTPUT_CLASSES];

    }

    public double[] fullyConnectedForwardPass(double[] input) {
        validateInput(input);

        lastX = input;
        double[] z = biases.clone();

        for (int i = 0; i < inLength; i++) {
            double xi = input[i];
            if (xi != 0.0) { // если xi == 0, то все произведения будут 0
                double[] wRow = weights[i];
                for (int j = 0; j < OUTPUT_CLASSES; j++) {
                    z[j] += xi * wRow[j];
                }
            }
        }

        lastZ = z;
        return applyActivation(z);
    }

    private void validateInput(double[] input) {
        if (input.length != inLength) {
            throw new IllegalArgumentException("Expected input length " + inLength + ", got " + input.length);
        }
    }

    private double[] applyActivation(double[] z) {
        double[] out = new double[OUTPUT_CLASSES];
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            out[j] = activation.forward(z[j]);
        }
        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        return nextLayer == null ? forwardPass : nextLayer.getOutput(forwardPass);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        // 1) dZ = dL/dO ⊙ f'(Z)
        double[] dZ = new double[OUTPUT_CLASSES];
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            dZ[j] = dLdO[j] * activation.backward(lastZ[j]);
        }

        // 2) dL/dX = W * dZ  (используем СТАРЫЕ веса)
        double[] dLdX = new double[inLength];
        for (int i = 0; i < inLength; i++) {
            double sum = 0.0;
            double[] wRow = weights[i];
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                sum += wRow[j] * dZ[j];
            }
            dLdX[i] = sum;
        }

        // 3) обновление весов: dL/dW_ij = X_i * dZ_j
        for (int i = 0; i < inLength; i++) {
            double xi = lastX[i];
            double[] wRow = weights[i];
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                wRow[j] -= learningRate * (xi * dZ[j]);
            }
        }

        // 4) обновление bias: dL/db_j = dZ_j
        for (int j = 0; j < OUTPUT_CLASSES; j++) {
            biases[j] -= learningRate * dZ[j];
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return OUTPUT_CLASSES;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return OUTPUT_CLASSES;
    }

    @Override
    public int getParameterCount() {
        return inLength * OUTPUT_CLASSES + OUTPUT_CLASSES;  // Weight matrix size + bias
    }

    @Override
    public String toString() {
        return String.format("🔗 FULLY CONNECTED | Inputs: %d → Outputs: %d | Parameters: %d",
            inLength, OUTPUT_CLASSES, getParameterCount());
    }


    private void initWeightsHe() {
        double std = Math.sqrt(2.0 / inLength);
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                weights[i][j] = RANDOM.nextGaussian() * std;
            }
        }
    }

    private void initWeightsXavier() {
        double limit = Math.sqrt(6.0 / (inLength + OUTPUT_CLASSES));
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                weights[i][j] = (RANDOM.nextDouble() * 2 - 1) * limit;
            }
        }
    }

}
