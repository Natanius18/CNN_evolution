package natanius.thesis.cnn.evolution.layers;

import java.util.ArrayList;
import java.util.List;
import lombok.Setter;

@Setter
public abstract class Layer {

    protected Layer nextLayer;
    protected Layer previousLayer;

    // ========== BATCH INTERFACE ==========
    // Теперь основные методы работают со списками (батчами)

    /**
     * Forward pass для батча входов
     * @param batchInput список входов для батча, где каждый элемент — List<double[][]> (каналы одного изображения)
     * @return список выходов для батча (List<double[]> или что требует конкретный слой)
     */
    public abstract List<double[]> getOutputBatch(List<List<double[][]>> batchInput);

    /**
     * Backpropagation для батча градиентов
     * @param dLdOBatch список градиентов для батча
     */
    public abstract void backPropagationBatch(List<double[]> dLdOBatch);

    // ========== METADATA ==========
    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();
    public abstract int getParameterCount();

    // ========== UTILITY METHODS ==========

    /**
     * Конвертирует список матриц (каналы) в вектор
     */
    protected double[] matrixToVector(List<double[][]> input) {
        if (input.isEmpty()) return new double[0];

        double[][] first = input.get(0);
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
     * Конвертирует вектор в список матриц (каналы)
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
