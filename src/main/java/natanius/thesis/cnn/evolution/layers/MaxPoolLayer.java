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

    // Batch storage для backpropagation
    // lastMaxRowBatch[b][r][c] = строка максимума для примера b, позиции (r,c)
    // lastMaxColBatch[b][r][c] = колонка максимума для примера b, позиции (r,c)
    private List<List<int[][]>> lastMaxRowBatch;
    private List<List<int[][]>> lastMaxColBatch;

    // ========== BATCH FORWARD PASS ==========

    @Override
    public List<double[]> getOutputBatch(List<List<double[][]>> batchInput) {
        List<List<double[][]>> pooledOutput = maxPoolForwardPassBatch(batchInput);

        // Конвертируем в векторы для следующего слоя
        List<double[]> batchVectors = new ArrayList<>();
        for (List<double[][]> featureMaps : pooledOutput) {
            batchVectors.add(matrixToVector(featureMaps));
        }

        // Передаём следующему слою, если есть
        if (nextLayer != null) {
            return nextLayer.getOutputBatch(pooledOutput);
        }
        return batchVectors;
    }

    /**
     * Forward pass для батча изображений
     * Каждое изображение содержит список каналов (feature maps)
     */
    public List<List<double[][]>> maxPoolForwardPassBatch(List<List<double[][]>> batchInputs) {
        List<List<double[][]>> batchOutputs = new ArrayList<>();
        lastMaxRowBatch = new ArrayList<>();
        lastMaxColBatch = new ArrayList<>();

        for (List<double[][]> input : batchInputs) {
            List<double[][]> channelOutputs = new ArrayList<>();
            List<int[][]> channelMaxRows = new ArrayList<>();
            List<int[][]> channelMaxCols = new ArrayList<>();

            // Pooling для каждого канала
            for (double[][] channel : input) {
                double[][] pooledChannel = new double[getOutputRows()][getOutputCols()];
                int[][] maxRows = new int[getOutputRows()][getOutputCols()];
                int[][] maxCols = new int[getOutputRows()][getOutputCols()];

                // Процесс pooling
                for (int r = 0; r < getOutputRows(); r++) {
                    for (int c = 0; c < getOutputCols(); c++) {
                        double max = Double.NEGATIVE_INFINITY;
                        int maxRowIdx = -1;
                        int maxColIdx = -1;

                        int startRow = r * stepSize;
                        int startCol = c * stepSize;

                        // Ищем максимум в окне
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

    // ========== BATCH BACKPROPAGATION ==========

    @Override
    public void backPropagationBatch(List<double[]> dLdOBatch) {
        // Конвертируем векторы обратно в feature maps
        List<List<double[][]>> dLdOFeatureMapsBatch = new ArrayList<>();
        for (double[] vec : dLdOBatch) {
            dLdOFeatureMapsBatch.add(vectorToMatrix(vec, getOutputLength(), getOutputRows(), getOutputCols()));
        }

        backPropagationBatchInternal(dLdOFeatureMapsBatch);
    }

    /**
     * Backpropagation для батча градиентов
     * Для каждого примера пропускаем ошибку через позиции максимумов
     */
    private void backPropagationBatchInternal(List<List<double[][]>> dLdOBatch) {
        int batchSize = dLdOBatch.size();
        List<List<double[][]>> dLdOPrevBatch = new ArrayList<>();

        for (int b = 0; b < batchSize; b++) {
            List<double[][]> dLdO = dLdOBatch.get(b);  // список градиентов по каналам
            List<int[][]> maxRows = lastMaxRowBatch.get(b);
            List<int[][]> maxCols = lastMaxColBatch.get(b);

            List<double[][]> dLdXChannels = new ArrayList<>();

            // Обрабатываем каждый канал
            for (int c = 0; c < inLength; c++) {
                double[][] gradOutput = dLdO.get(c);  // градиент для этого канала
                int[][] maxRowIdx = maxRows.get(c);
                int[][] maxColIdx = maxCols.get(c);

                // Восстанавливаем градиент до pooling
                double[][] gradInput = new double[inRows][inCols];

                for (int r = 0; r < getOutputRows(); r++) {
                    for (int col = 0; col < getOutputCols(); col++) {
                        int maxI = maxRowIdx[r][col];
                        int maxJ = maxColIdx[r][col];

                        // Ошибка пропускается только в позицию максимума
                        if (maxI != -1 && maxJ != -1) {
                            gradInput[maxI][maxJ] += gradOutput[r][col];
                        }
                    }
                }

                dLdXChannels.add(gradInput);
            }

            dLdOPrevBatch.add(dLdXChannels);
        }

        // Передаём батч градиентов предыдущему слою
        if (previousLayer != null) {
            List<double[]> dLdOPrevVectors = new ArrayList<>();
            for (List<double[][]> featureMaps : dLdOPrevBatch) {
                dLdOPrevVectors.add(matrixToVector(featureMaps));
            }
            previousLayer.backPropagationBatch(dLdOPrevVectors);
        }
    }

    // ========== METADATA ==========

    @Override
    public int getOutputLength() {
        return inLength;  // Количество каналов не меняется
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
        return 0;  // Нет обучаемых параметров в pooling
    }

    @Override
    public String toString() {
        return String.format("🔄 MAX POOL | Window: %dx%d | Stride: %d | Input: %dx%d | Output: %dx%d",
            windowSize, windowSize, stepSize, inRows, inCols, getOutputRows(), getOutputCols());
    }
}
