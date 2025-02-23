package natanius.thesis.cnnEvolution.layers;

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

    public double[] matrixToVector(List<double[][]> input) {

        int length = input.size();
        int rows = input.getFirst().length;
        int cols = input.getFirst()[0].length;

        double[] vector = new double[length * rows * cols];

        int i = 0;
        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }

        return vector;
    }

    List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {

        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for (int l = 0; l < length; l++) {

            double[][] matrix = new double[rows][cols];

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = input[i];
                    i++;
                }
            }
            out.add(matrix);
        }

        return out;
    }

}
