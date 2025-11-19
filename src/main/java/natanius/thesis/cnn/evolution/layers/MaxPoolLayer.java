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

    private List<List<int[][]>> lastMaxRowBatch;
    private List<List<int[][]>> lastMaxColBatch;


    @Override
    public List<double[]> getOutputBatch(List<List<double[][]>> batchInput) {
        List<List<double[][]>> pooledOutput = maxPoolForwardPassBatch(batchInput);

        // Конвертуємо в вектори для наступного шару
        List<double[]> batchVectors = new ArrayList<>();
        for (List<double[][]> featureMaps : pooledOutput) {
            batchVectors.add(matrixToVector(featureMaps));
        }

        // Передаємо наступному шару, якщо є
        if (nextLayer != null) {
            return nextLayer.getOutputBatch(pooledOutput);
        }
        return batchVectors;
    }

    /**
     * Виконує forward pass max pooling для батча вхідних feature maps.
     * <p>
     * Max pooling зменшує просторові розміри, зберігаючи найважливіші ознаки.
     * Для кожного вікна розміром windowSize×windowSize обирається максимальне значення.
     * <p>
     * <b>Важливо:</b> Зберігаються позиції максимумів для backpropagation.
     *
     * @param batchInputs список вхідних feature maps [batchSize][inLength][inRows][inCols]
     * @return список вихідних feature maps [batchSize][inLength][outRows][outCols]
     */
    public List<List<double[][]>> maxPoolForwardPassBatch(List<List<double[][]>> batchInputs) {
        List<List<double[][]>> batchOutputs = new ArrayList<>();
        lastMaxRowBatch = new ArrayList<>();
        lastMaxColBatch = new ArrayList<>();

        for (List<double[][]> input : batchInputs) {
            List<double[][]> channelOutputs = new ArrayList<>();
            List<int[][]> channelMaxRows = new ArrayList<>();
            List<int[][]> channelMaxCols = new ArrayList<>();

            // Pooling для кожного каналу
            for (double[][] channel : input) {
                double[][] pooledChannel = new double[getOutputRows()][getOutputCols()];
                int[][] maxRows = new int[getOutputRows()][getOutputCols()];
                int[][] maxCols = new int[getOutputRows()][getOutputCols()];

                // Процес pooling
                for (int r = 0; r < getOutputRows(); r++) {
                    for (int c = 0; c < getOutputCols(); c++) {
                        double max = Double.NEGATIVE_INFINITY;
                        int maxRowIdx = -1;
                        int maxColIdx = -1;

                        int startRow = r * stepSize;
                        int startCol = c * stepSize;

                        // Шукаємо максимум у вікні
                        for (int x = 0; x < windowSize; x++) {
                            for (int y = 0; y < windowSize; y++) {
                                double value = channel[startRow + x][startCol + y];
                                if (value > max) {
                                    max = value;
                                    maxRowIdx = startRow + x;
                                    maxColIdx = startCol + y;
                                }
                            }
                        }

                        pooledChannel[r][c] = max;
                        maxRows[r][c] = maxRowIdx;
                        maxCols[r][c] = maxColIdx;
                    }
                }

                channelOutputs.add(pooledChannel);
                channelMaxRows.add(maxRows);
                channelMaxCols.add(maxCols);
            }

            batchOutputs.add(channelOutputs);
            lastMaxRowBatch.add(channelMaxRows);
            lastMaxColBatch.add(channelMaxCols);
        }

        return batchOutputs;
    }


    @Override
    public void backPropagationBatch(List<double[]> dLdOBatch) {
        // Конвертуємо вектори назад у feature maps
        List<List<double[][]>> dLdOFeatureMapsBatch = new ArrayList<>();
        for (double[] vec : dLdOBatch) {
            dLdOFeatureMapsBatch.add(vectorToMatrix(vec, getOutputLength(), getOutputRows(), getOutputCols()));
        }

        backPropagationBatchInternal(dLdOFeatureMapsBatch);
    }

    /**
     * Виконує backpropagation через max pooling шар для батча градієнтів.
     * <p>
     * Max pooling не має параметрів для навчання, тому градієнт просто
     * передається назад тільки в ті позиції, де були максимальні значення
     * під час forward pass. Всі інші позиції отримують градієнт 0.
     * <p>
     * <b>Математична операція:</b>
     * <pre>
     * ∂L/∂x[i][j] = ∂L/∂y[r][c], якщо x[i][j] був максимумом у вікні (r,c)
     * ∂L/∂x[i][j] = 0, інакше
     * </pre>
     *
     * @param dLdOBatch список градієнтів виходу [batchSize][inLength][outRows][outCols]
     */
    private void backPropagationBatchInternal(List<List<double[][]>> dLdOBatch) {
        int batchSize = dLdOBatch.size();
        List<List<double[][]>> dLdOPrevBatch = new ArrayList<>();

        for (int b = 0; b < batchSize; b++) {
            List<double[][]> dLdO = dLdOBatch.get(b);  // список градієнтів по каналах
            List<int[][]> maxRows = lastMaxRowBatch.get(b);
            List<int[][]> maxCols = lastMaxColBatch.get(b);

            List<double[][]> dLdXChannels = new ArrayList<>();

            // Обробляємо кожен канал
            for (int c = 0; c < inLength; c++) {
                double[][] gradOutput = dLdO.get(c);  // градієнт для цього каналу
                int[][] maxRowIdx = maxRows.get(c);
                int[][] maxColIdx = maxCols.get(c);

                // Відновлюємо градієнт до pooling
                double[][] gradInput = new double[inRows][inCols];

                for (int r = 0; r < getOutputRows(); r++) {
                    for (int col = 0; col < getOutputCols(); col++) {
                        int maxI = maxRowIdx[r][col];
                        int maxJ = maxColIdx[r][col];

                        // Помилка передається тільки в позицію максимуму
                        if (maxI != -1 && maxJ != -1) {
                            gradInput[maxI][maxJ] += gradOutput[r][col];
                        }
                    }
                }

                dLdXChannels.add(gradInput);
            }

            dLdOPrevBatch.add(dLdXChannels);
        }

        // Передаємо батч градієнтів попередньому шару
        if (previousLayer != null) {
            List<double[]> dLdOPrevVectors = new ArrayList<>();
            for (List<double[][]> featureMaps : dLdOPrevBatch) {
                dLdOPrevVectors.add(matrixToVector(featureMaps));
            }
            previousLayer.backPropagationBatch(dLdOPrevVectors);
        }
    }


    @Override
    public int getOutputLength() {
        return inLength;  // Кількість каналів не змінюється
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
        return 0;  // Немає параметрів для навчання в pooling
    }

    @Override
    public String toString() {
        return String.format("🔄 MAX POOL | Window: %dx%d | Stride: %d | Input: %dx%d | Output: %dx%d",
            windowSize, windowSize, stepSize, inRows, inCols, getOutputRows(), getOutputCols());
    }
}
