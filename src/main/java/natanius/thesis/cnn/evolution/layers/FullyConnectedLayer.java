package natanius.thesis.cnn.evolution.layers;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer implements Serializable {

    private final long seed;
    private final double leak = 0.01;

    private final double[][] weights;
    private final int inLength;
    private final int outLength;
    private final double learningRate;

    private double[] lastZ;
    private double[] lastX;


    public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        this.inLength = inLength;
        this.outLength = outLength;
        this.seed = seed;
        this.learningRate = learningRate;

        weights = new double[inLength][outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input){

        lastX = input;

        double[] z = new double[outLength];
        double[] out = new double[outLength];

        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
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

            for (int j = 0; j < outLength; j++) {

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = weights[k][j];

                dLdw = dLdO[j]*dOdz*dzdw;

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
        return outLength;
    }

    @Override
    public int getParameterCount() {
        return inLength * outLength;  // Weight matrix size
    }

    @Override
    public String toString() {
        return String.format("🔗 FULLY CONNECTED | Inputs: %d → Outputs: %d | Parameters: %d",
            inLength, outLength, getParameterCount());
    }



    public void setRandomWeights(){
        Random random = new Random(seed);

        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double reLu(double input){
        return input <= 0 ? 0 : input;
    }

    public double derivativeReLu(double input){
        return input <= 0 ? leak : 1;
    }

}
