package natanius.thesis.cnn.evolution.layers;

import java.util.ArrayList;
import java.util.List;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class MaxPoolLayer extends Layer {

    private final int stepSize;
    private final int windowSize;

    private final int inLength;
    private final int inRows;
    private final int inCols;

    private List<int[][]> lastMaxRow;
    private List<int[][]> lastMaxCol;

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        lastMaxRow = new ArrayList<>();
        lastMaxCol = new ArrayList<>();

        for (double[][] doubles : input) {
            output.add(pool(doubles));
        }

        return output;

    }

    public double[][] pool(double[][] input) {

        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for (int r = 0; r < getOutputRows(); r += stepSize) {
            for (int c = 0; c < getOutputCols(); c += stepSize) {

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for (int x = 0; x < windowSize; x++) {
                    for (int y = 0; y < windowSize; y++) {
                        if (max < input[r + x][c + y]) {
                            max = input[r + x][c + y];
                            maxRows[r][c] = r + x;
                            maxCols[r][c] = c + y;
                        }
                    }
                }

                output[r][c] = max;

            }
        }

        lastMaxRow.add(maxRows);
        lastMaxCol.add(maxCols);

        return output;

    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, inLength, inRows, inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> dXdL = new ArrayList<>();

        int l = 0;
        for (double[][] array : dLdO) {
            double[][] error = new double[inRows][inCols];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int maxi = lastMaxRow.get(l)[r][c];
                    int maxj = lastMaxCol.get(l)[r][c];

                    if (maxi != -1) {
                        error[maxi][maxj] += array[r][c];
                    }
                }
            }

            dXdL.add(error);
            l++;
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dXdL);
        }

    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength * getOutputCols() * getOutputRows();
    }

    @Override
    public int getParameterCount() {
        return 0;  // No trainable parameters in pooling layers
    }

    @Override
    public String toString() {
        return String.format("🔄 MAX POOL | Window: %dx%d | Stride: %d",
            windowSize, windowSize, stepSize);
    }

}
