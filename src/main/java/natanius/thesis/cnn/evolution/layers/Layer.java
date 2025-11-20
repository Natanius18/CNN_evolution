package natanius.thesis.cnn.evolution.layers;

import java.util.ArrayList;
import java.util.List;
import lombok.Setter;

@Setter
public abstract class Layer {

    protected Layer nextLayer;
    protected Layer previousLayer;


    /**
     * Forward pass для батча входів
     * @param batchInput список входів для батча, де кожен елемент — List<double[][]> (канали одного зображення)
     * @return список виходів для батча (List<double[]> або що вимагає конкретний шар)
     */
    public abstract List<double[]> getOutputBatch(List<List<double[][]>> batchInput);

    /**
     * Backpropagation для батча градієнтів
     * @param dLdOBatch список градієнтів для батча
     */
    public abstract void backPropagationBatch(List<double[]> dLdOBatch);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();
    public abstract int getParameterCount();


    /**
     * Конвертує список матриць (канали) у вектор
     */
    protected double[] matrixToVector(List<double[][]> input) {
        if (input.isEmpty()) return new double[0];

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

    /**
     * Конвертує вектор у список матриць (канали)
     */
    protected List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
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
