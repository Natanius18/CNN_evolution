package natanius.thesis.cnn.evolution.layers;

import static natanius.thesis.cnn.evolution.data.Constants.L2_REGULARIZATION_LAMBDA;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.data.MatrixUtility.add;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.activation.LeakyReLU;
import natanius.thesis.cnn.evolution.activation.ReLU;
import natanius.thesis.cnn.evolution.activation.Sigmoid;

public class ConvolutionLayer extends Layer {

    private final int filterSize;
    private final int stepSize;
    private final int padding;
    private final int inLength;
    private final int inRows;
    private final int inCols;
    private final double learningRate;
    private final List<double[][][]> filters = new ArrayList<>();
    private final Activation activation;
    private final double[] biases;
    private double l2Lambda = L2_REGULARIZATION_LAMBDA;
    private List<List<double[][]>> lastInputBatch;
    private List<List<double[][]>> preActivationOutputsBatch;

    public ConvolutionLayer(int filterSize,
                            int stepSize,
                            int padding,
                            int inLength,
                            int inRows,
                            int inCols,
                            int numFilters,
                            double learningRate,
                            Activation activation) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.padding = padding;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
        this.learningRate = learningRate;
        this.activation = activation;
        this.biases = new double[numFilters];

        generateRandomFilters(numFilters);
    }


    @Override
    public List<double[]> getOutputBatch(List<List<double[][]>> batchInput) {
        List<List<double[][]>> batchFeatureMaps = convolutionForwardPassBatch(batchInput);

        List<double[]> batchVectors = new ArrayList<>();
        for (List<double[][]> featureMaps : batchFeatureMaps) {
            batchVectors.add(matrixToVector(featureMaps));
        }

        if (nextLayer != null) {
            return nextLayer.getOutputBatch(batchFeatureMaps);
        }
        return batchVectors;
    }

    /**
     * Виконує forward pass згорткового шару для батчу вхідних feature maps.
     * <p>
     * Кожен фільтр застосовується до всіх вхідних каналів, створюючи одну вихідну feature map.
     * Якщо вхід містить N каналів, а шар має M фільтрів, то вихід міститиме M feature maps.
     * <p><b>Математична операція:</b> Для кожного фільтра виконується
     * дискретна згортка по всіх вхідних каналах:
     * <pre>
     * Output[i][j] = Σ Σ Σ Input[c][i×stride + x][j×stride + y] × Filter[c][x][y]
     *                c x y
     * </pre>
     * <b>Важливо:</b> Метод зберігає вхідні дані в {@code lastInputBatch} для
     * використання під час backpropagation.
     *
     * @param batchInputs список вхідних feature maps розміром [batchSize][inLength][inRows][inCols]
     * @return список вихідних feature maps після згортки, розмір:
     * <p>
     * [batchSize][numFilters][outRows][outCols],
     * <p>
     * де outRows та outCols визначаються формулою:
     * <p>
     * (size + 2×padding - filterSize) / stride + 1
     */
    public List<List<double[][]>> convolutionForwardPassBatch(List<List<double[][]>> batchInputs) {
        List<List<double[][]>> batchOutputs = new ArrayList<>();
        lastInputBatch = new ArrayList<>();
        preActivationOutputsBatch = new ArrayList<>();

        for (List<double[][]> input : batchInputs) {
            if (input.size() != inLength) {
                throw new IllegalArgumentException(
                    "Expected " + inLength + " input channels, got " + input.size()
                );
            }

            List<double[][]> preActivationOutputs = new ArrayList<>();
            List<double[][]> output = new ArrayList<>();

            // Кожен фільтр створює одну feature map
            for (int f = 0; f < filters.size(); f++) {
                double[][] featureMap = convolveMultiChannel(input, filters.get(f), f, preActivationOutputs);
                output.add(featureMap);
            }

            batchOutputs.add(output);
            lastInputBatch.add(cloneListOfMatrices(input));
            preActivationOutputsBatch.add(cloneListOfMatrices(preActivationOutputs));
        }

        return batchOutputs;
    }

    /**
     * Виконує multi-channel згортку: один фільтр застосовується до всіх вхідних каналів.
     * Результати підсумовуються.
     */
    private double[][] convolveMultiChannel(List<double[][]> inputs,
                                            double[][][] filter,
                                            int filterIndex,
                                            List<double[][]> preActivationOutputs) {
        double[][][] paddedInputs = new double[inLength][][];
        for (int c = 0; c < inLength; c++) {
            paddedInputs[c] = applyPadding(inputs.get(c));
        }

        int paddedRows = paddedInputs[0].length;
        int paddedCols = paddedInputs[0][0].length;
        int outRows = (paddedRows - filterSize) / stepSize + 1;
        int outCols = (paddedCols - filterSize) / stepSize + 1;

        double[][] preActivationOutput = new double[outRows][outCols];
        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        for (int i = 0; i <= paddedRows - filterSize; i += stepSize) {
            int outCol = 0;
            for (int j = 0; j <= paddedCols - filterSize; j += stepSize) {
                double sum = 0.0;

                // Підсумовуємо по всіх каналах
                for (int c = 0; c < inLength; c++) {
                    for (int x = 0; x < filterSize; x++) {
                        for (int y = 0; y < filterSize; y++) {
                            sum += paddedInputs[c][i + x][j + y] * filter[c][x][y];
                        }
                    }
                }

                sum += biases[filterIndex];
                preActivationOutput[outRow][outCol] = sum;
                output[outRow][outCol] = activation.forward(sum);
                outCol++;
            }
            outRow++;
        }

        preActivationOutputs.add(preActivationOutput);
        return output;
    }


    @Override
    public void backPropagationBatch(List<double[]> dLdOBatch) {
        // Конвертуємо вектори в feature maps
        List<List<double[][]>> dLdOFeatureMapsBatch = new ArrayList<>();
        for (double[] vec : dLdOBatch) {
            dLdOFeatureMapsBatch.add(vectorToMatrix(vec, getOutputLength(), getOutputRows(), getOutputCols()));
        }

        backPropagationBatchInternal(dLdOFeatureMapsBatch);
    }

    /**
     * Backpropagation для батча градієнтів
     */
    private void backPropagationBatchInternal(List<List<double[][]>> dLdOBatch) {
        int batchSize = dLdOBatch.size();

        // Ініціалізація акумуляторів градієнтів
        List<double[][][]> filtersDeltaSum = new ArrayList<>();
        for (int f = 0; f < filters.size(); f++) {
            filtersDeltaSum.add(new double[inLength][filterSize][filterSize]);
        }
        double[] biasesDeltaSum = new double[filters.size()];

        List<List<double[][]>> dLdOPrevBatch = new ArrayList<>();

        // Обробляємо кожен приклад у батчі
        for (int b = 0; b < batchSize; b++) {
            List<double[][]> dLdO = dLdOBatch.get(b);
            List<double[][]> lastInput = lastInputBatch.get(b);
            List<double[][]> preAct = preActivationOutputsBatch.get(b);

            // КРОК 1: Градієнт через activation
            List<double[][]> dLdZ = new ArrayList<>();
            for (int idx = 0; idx < dLdO.size(); idx++) {
                double[][] gradOutput = dLdO.get(idx);
                double[][] preActivation = preAct.get(idx);
                int rows = gradOutput.length;
                int cols = gradOutput[0].length;
                double[][] gradPreActivation = new double[rows][cols];

                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        gradPreActivation[r][c] = gradOutput[r][c] * activation.backward(preActivation[r][c]);
                    }
                }
                dLdZ.add(gradPreActivation);
            }

            // КРОК 2: Обчислення градієнтів
            List<double[][][]> filtersDelta = new ArrayList<>();
            for (int f = 0; f < filters.size(); f++) {
                filtersDelta.add(new double[inLength][filterSize][filterSize]);
            }
            double[] biasesDelta = new double[filters.size()];

            // Градієнт по входу окремо для кожного каналу
            List<double[][]> dLdOPreviousLayer = new ArrayList<>();
            for (int c = 0; c < inLength; c++) {
                dLdOPreviousLayer.add(new double[inRows][inCols]);
            }

            // Проходимо по кожному фільтру
            for (int f = 0; f < filters.size(); f++) {
                double[][][] currFilter = filters.get(f);
                double[][] error = dLdZ.get(f);
                double[][] spacedError = spaceArray(error);
                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));

                // Градієнт bias
                for (double[] row : error) {
                    for (double val : row) {
                        biasesDelta[f] += val;
                    }
                }

                // По кожному каналу окремо
                for (int c = 0; c < inLength; c++) {
                    double[][] paddedInput = applyPadding(lastInput.get(c));
                    double[][] dLdF = pureConvolve(paddedInput, flippedError);
                    add(filtersDelta.get(f)[c], dLdF);

                    // Градієнт по входу (канал c)
                    double[][] flippedFilter = flipArrayHorizontal(flipArrayVertical(currFilter[c]));
                    double[][] convResult = fullConvolve(flippedFilter, spacedError);

                    // Обрізаємо до розміру входу
                    convResult = cropToSize(convResult, inRows, inCols, padding);
                    add(dLdOPreviousLayer.get(c), convResult);
                }
            }

            // Акумулюємо градієнти по батчу
            for (int f = 0; f < filters.size(); f++) {
                for (int c = 0; c < inLength; c++) {
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            filtersDeltaSum.get(f)[c][i][j] += filtersDelta.get(f)[c][i][j];
                        }
                    }
                }
                biasesDeltaSum[f] += biasesDelta[f];
            }

            dLdOPrevBatch.add(dLdOPreviousLayer);
        }

        // КРОК 3: Оновлення ваг (усереднено по батчу)
        for (int f = 0; f < filters.size(); f++) {
            for (int c = 0; c < inLength; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        double grad = filtersDeltaSum.get(f)[c][i][j] / batchSize;
                        grad += l2Lambda * filters.get(f)[c][i][j];  // L2 регуляризація
                        filters.get(f)[c][i][j] -= learningRate * grad;
                    }
                }
            }
            biases[f] -= learningRate * (biasesDeltaSum[f] / batchSize);
        }

        // Передаємо градієнти попередньому шару
        if (previousLayer != null) {
            List<double[]> dLdOPrevVectors = new ArrayList<>();
            for (List<double[][]> featureMaps : dLdOPrevBatch) {
                dLdOPrevVectors.add(matrixToVector(featureMaps));
            }
            previousLayer.backPropagationBatch(dLdOPrevVectors);
        }
    }


    private List<double[][]> cloneListOfMatrices(List<double[][]> list) {
        List<double[][]> clone = new ArrayList<>();
        for (double[][] mat : list) {
            double[][] matCopy = new double[mat.length][];
            for (int i = 0; i < mat.length; i++) {
                matCopy[i] = mat[i].clone();
            }
            clone.add(matCopy);
        }
        return clone;
    }


    /**
     * Додає zero padding (обрамлення з нулів) навколо вхідної матриці.
     *
     * <p>Padding використовується для:
     * <ul>
     *   <li>Збереження просторових розмірів після згортки (same padding)</li>
     *   <li>Запобігання втраті інформації з країв зображення</li>
     *   <li>Контролю розміру вихідних feature maps</li>
     * </ul>
     *
     * <p><b>Приклад:</b> padding = 1
     * <pre>
     * [1 2]     =>    [0 0 0 0]
     * [3 4]           [0 1 2 0]
     *                 [0 3 4 0]
     *                 [0 0 0 0]
     * </pre>
     *
     * @param input вхідна матриця розміром [rows][cols]
     * @return матриця розміром [rows + 2×padding][cols + 2×padding] з доданим padding
     */
    private double[][] applyPadding(double[][] input) {
        if (padding == 0) return input;

        int inRows = input.length;
        int inCols = input[0].length;
        int paddedRows = inRows + 2 * padding;
        int paddedCols = inCols + 2 * padding;

        double[][] padded = new double[paddedRows][paddedCols];
        for (int i = 0; i < inRows; i++) {
            System.arraycopy(input[i], 0, padded[i + padding], padding, inCols);
        }
        return padded;
    }

    private double[][] cropToSize(double[][] input, int targetRows, int targetCols, int padding) {
        double[][] output = new double[targetRows][targetCols];
        int startRow = padding;
        int startCol = padding;

        for (int i = 0; i < targetRows && (startRow + i) < input.length; i++) {
            for (int j = 0; j < targetCols && (startCol + j) < input[0].length; j++) {
                output[i][j] = input[startRow + i][startCol + j];
            }
        }
        return output;
    }

    /**
     * Виконує spacing операцію (zero interweaving) для матриці градієнтів під час backpropagation.
     *
     * <p>Якщо під час forward pass використовувався stride > 1, то градієнти отримані після
     * згортки мають зменшений розмір. Цей метод вставляє нулі між елементами градієнта,
     * щоб відновити правильний spacing для подальшого обчислення градієнтів попереднього шару.
     *
     * <p><b>Приклад:</b> Для stride = 2 та вхідної матриці:
     * <pre>
     * [1 2]  =>  [1 0 2]
     * [3 4]      [0 0 0]
     *            [3 0 4]
     * </pre>
     *
     * <p><b>Формула розміру виходу:</b>
     * <ul>
     *   <li>outRows = (inputRows - 1) × stride + 1</li>
     *   <li>outCols = (inputCols - 1) × stride + 1</li>
     * </ul>
     *
     * @param input вхідна матриця градієнтів після згортки
     * @return матриця з вставленими нулями між елементами, якщо stride > 1;
     * <p>
     * незмінена матриця, якщо stride = 1
     */
    private double[][] spaceArray(double[][] input) {
        if (stepSize == 1) return input;

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length - 1) * stepSize + 1;
        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * stepSize][j * stepSize] = input[i][j];
            }
        }
        return output;
    }

    private double[][] flipArrayVertical(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(array[i], 0, output[rows - i - 1], 0, cols);
        }
        return output;
    }

    private double[][] flipArrayHorizontal(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }
        return output;
    }

    /**
     * Виконує чисту операцію згортки БЕЗ застосування активаційної функції та БЕЗ padding.
     * Використовується під час backpropagation для обчислення градієнтів фільтрів.
     */
    private double[][] pureConvolve(double[][] input, double[][] filter) {
        int inRows = input.length;
        int inCols = input[0].length;
        int fRows = filter.length;
        int fCols = filter[0].length;
        int outRows = (inRows - fRows) + 1;
        int outCols = (inCols - fCols) + 1;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        for (int i = 0; i <= inRows - fRows; i++) {
            int outCol = 0;
            for (int j = 0; j <= inCols - fCols; j++) {
                double sum = 0.0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        sum += input[i + x][j + y] * filter[x][y];
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }
        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int inRows = input.length;
        int inCols = input[0].length;
        int fRows = filter.length;
        int fCols = filter[0].length;
        int outRows = inRows + fRows - 1;
        int outCols = inCols + fCols - 1;

        double[][] output = new double[outRows][outCols];

        for (int i = 0; i < outRows; i++) {
            for (int j = 0; j < outCols; j++) {
                double sum = 0.0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inRow = i - x;
                        int inCol = j - y;
                        if (inRow >= 0 && inCol >= 0 && inRow < inRows && inCol < inCols) {
                            sum += filter[x][y] * input[inRow][inCol];
                        }
                    }
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    /**
     * Генерує випадкові фільтри з адаптивною ініціалізацією залежно від типу активації.
     *
     * <p>Використовує:
     * <ul>
     *   <li>He initialization для ReLU та LeakyReLU</li>
     *   <li>Xavier initialization для Sigmoid та Tanh</li>
     * </ul>
     *
     * @param numFilters кількість фільтрів для генерації
     */
    private void generateRandomFilters(int numFilters) {
        if (activation instanceof ReLU || activation instanceof LeakyReLU) {
            initFiltersHe(numFilters);
        } else if (activation instanceof Sigmoid) {
            initFiltersXavier(numFilters);
        } else {
            throw new IllegalArgumentException(
                "Unsupported activation function: " + activation.getClass().getSimpleName()
            );
        }
    }

    /**
     * Ініціалізує фільтри за методом He (Kaiming) для ReLU та LeakyReLU активацій.
     *
     * <p>He initialization оптимізована для ReLU, яка обнуляє негативні значення.
     * <p><b>Формула:</b> W ~ N(0, sqrt(2 / fan_in)), де fan_in = filterSize² × inLength
     *
     * @param numFilters кількість фільтрів для створення
     * @see <a href="https://arxiv.org/abs/1502.01852">He et al., 2015</a>
     */
    private void initFiltersHe(int numFilters) {
        int fanIn = filterSize * filterSize * inLength;
        double std = Math.sqrt(2.0 / fanIn);

        for (int n = 0; n < numFilters; n++) {
            double[][][] newFilter = new double[inLength][filterSize][filterSize];
            for (int c = 0; c < inLength; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        newFilter[c][i][j] = RANDOM.nextGaussian() * std;
                    }
                }
            }
            filters.add(newFilter);
        }
    }

    /**
     * Ініціалізує фільтри за методом Xavier (Glorot) для Sigmoid та Tanh активацій.
     *
     * <p>Xavier initialization оптимізована для симетричних активаційних функцій.
     * <p><b>Формула:</b> W ~ U(-limit, +limit), де limit = sqrt(6 / (fan_in + fan_out))
     *
     * @param numFilters кількість фільтрів для створення
     * @see <a href="http://proceedings.mlr.press/v9/glorot10a.html">Glorot & Bengio, 2010</a>
     */
    private void initFiltersXavier(int numFilters) {
        int fanIn = filterSize * filterSize * inLength;
        int fanOut = getOutputRows() * getOutputCols() * numFilters;
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));

        for (int n = 0; n < numFilters; n++) {
            double[][][] newFilter = new double[inLength][filterSize][filterSize];
            for (int c = 0; c < inLength; c++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        newFilter[c][i][j] = (RANDOM.nextDouble() * 2 - 1) * limit;
                    }
                }
            }
            filters.add(newFilter);
        }
    }

    @Override
    public int getOutputLength() {
        return filters.size();
    }

    /**
     * Для обчислення розміру виходу після операції згортки використовується формула:
     * <p>
     * H_out = (H_in - k + 2p) / s + 1,
     * <p>
     * де H_in — розміри входу, k — розмір ядра, p — padding, s — stride (крок зсуву фільтра).
     */
    @Override
    public int getOutputRows() {
        return (inRows - filterSize + 2 * padding) / stepSize + 1;
    }

    /**
     * W_out = (W_in - k + 2p) / s + 1,
     */
    @Override
    public int getOutputCols() {
        return (inCols - filterSize + 2 * padding) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }

    @Override
    public int getParameterCount() {
        return filters.size() * filterSize * filterSize * inLength  // ваги
            + filters.size();                                       // biases
    }

    @Override
    public String toString() {
        return String.format("🌀 CONVOLUTION | %d filters | %dx%d kernel | Stride: %d | Padding: %d | Parameters: %d",
            filters.size(), filterSize, filterSize, stepSize, padding, getParameterCount());
    }
}
