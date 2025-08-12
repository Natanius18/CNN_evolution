package natanius.thesis.cnn.evolution.layers;

import static natanius.thesis.cnn.evolution.data.Constants.OUTPUT_CLASSES;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.io.Serializable;
import java.util.List;

public class FullyConnectedLayer extends Layer implements Serializable {

    private final double leak = 0.01;

    private final double[][] weights;
    private final int inLength;
    private final double learningRate;

    private double[] lastZ;
    private double[] lastX;


    public FullyConnectedLayer(int inLength, double learningRate) {
        this.inLength = inLength;
        this.learningRate = learningRate;

        weights = new double[inLength][OUTPUT_CLASSES];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input) {

        lastX = input;

        double[] z = new double[OUTPUT_CLASSES];
        double[] out = new double[OUTPUT_CLASSES];

        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < inLength; i++) {  // todo: fix math, delete useless code
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                out[j] = reLu(z[j]);
            }
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

        if (nextLayer != null) {
            return nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {

        double[] dLdX = new double[inLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for (int k = 0; k < inLength; k++) {

            double dLdXsum = 0;

            for (int j = 0; j < OUTPUT_CLASSES; j++) {

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = weights[k][j];

                dLdw = dLdO[j] * dOdz * dzdw;

                weights[k][j] -= dLdw * learningRate;

                dLdXsum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[k] = dLdXsum;
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
        return 0;
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
        return inLength * OUTPUT_CLASSES;  // Weight matrix size
    }

    @Override
    public String toString() {
        return String.format("🔗 FULLY CONNECTED | Inputs: %d → Outputs: %d | Parameters: %d",
            inLength, OUTPUT_CLASSES, getParameterCount());
    }


    public void setRandomWeights() {
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                weights[i][j] = RANDOM.nextGaussian();
            }
        }
    }

    public double reLu(double input) {
        return input <= 0 ? 0 : input;
    }

    public double derivativeReLu(double input) {
        return input <= 0 ? leak : 1;
    }

}
