package natanius.thesis.cnn.evolution.layers;

import java.util.ArrayList;
import java.util.List;
import lombok.Setter;

@Setter
public abstract class Layer {

    protected Layer nextLayer;
    protected Layer previousLayer;

    public abstract double[] getOutput(List<double[][]> input);

    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(double[] dLdO);

    public abstract void backPropagation(List<double[][]> dLdO);

    public abstract int getOutputLength();

    public abstract int getOutputRows();

    public abstract int getOutputCols();

    public abstract int getOutputElements();
    public abstract int getParameterCount();

    public double[] matrixToVector(List<double[][]> input) {
        double[][] first = input.getFirst();
        int rows = first.length;
        int cols = first[0].length;
        double[] vector = new double[input.size() * rows * cols];
        
        int i = 0;
        for (double[][] matrix : input) {
            for (double[] row : matrix) {
                System.arraycopy(row, 0, vector, i, cols);
                i += cols;
            }
        }
        return vector;
    }

    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>(length);

        int i = 0;
        for (int l = 0; l < length; l++) {
            double[][] matrix = new double[rows][];
            for (int r = 0; r < rows; r++) {
                matrix[r] = new double[cols];
                System.arraycopy(input, i, matrix[r], 0, cols);
                i += cols;
            }
            out.add(matrix);
        }
        return out;
    }

}
