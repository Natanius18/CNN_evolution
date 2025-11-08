package natanius.thesis.cnn.evolution.layers;

import static natanius.thesis.cnn.evolution.data.Constants.L2_REGULARIZATION_LAMBDA;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.data.MatrixUtility.add;
import static natanius.thesis.cnn.evolution.data.MatrixUtility.multiply;

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
    private final List<double[][][]> filters = new ArrayList<>();  // [numFilters][filterSize][filterSize][inLength]
    private List<double[][]> lastInput;
    private final Activation activation;
    private List<double[][]> preActivationOutputs;
    private final double[] biases;
    private double l2Lamda = L2_REGULARIZATION_LAMBDA;

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
                "Unsupported activation function: " + activation.getClass().getSimpleName() +
                    ". Supported: ReLU, LeakyReLU, Sigmoid"
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
            double[][][] newFilter = new double[inLength][filterSize][filterSize];  // 3D фільтр

            for (int c = 0; c < inLength; c++) {  // По каналах
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

            for (int c = 0; c < inLength; c++) {  // ✅ По каналах
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
    public double[] getOutput(double[] input) {

        List<double[][]> matrixInput = vectorToMatrix(input, inLength, inRows, inCols);

        return getOutput(matrixInput);
    }

    @Override
    public double[] getOutput(List<double[][]> input) {

        List<double[][]> output = convolutionForwardPass(input);

        return nextLayer.getOutput(output);

    }

    /**
     * Виконує forward pass згорткового шару для батчу вхідних feature maps.
     * <p>
     * Для кожної вхідної feature map застосовується кожен фільтр шару,
     * створюючи нові feature maps. Якщо вхід містить N feature maps, а шар
     * має M фільтрів, то вихід міститиме N × M feature maps.
     * <p><b>Математична операція:</b> Для кожної пари (вхід, фільтр) виконується
     * дискретна згортка:
     * <pre>
     * Output[i][j] = Σ Σ Input[i×stride + x][j×stride + y] × Filter[x][y]
     *                x y
     * </pre>
     * <b>Важливо:</b> Метод зберігає вхідні дані в {@code lastInput} для
     * використання під час backpropagation.
     *
     * @param list список вхідних feature maps розміром [inLength][inRows][inCols]
     * @return список вихідних feature maps після згортки, розмір:
     * <p>
     * [inLength × numFilters][outRows][outCols],
     * <p>
     * де outRows та outCols визначаються формулою:
     * <p>
     * (size + 2×padding - filterSize) / stride + 1
     */
    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        if (list.size() != inLength) {
            throw new IllegalArgumentException(
                "Expected " + inLength + " input channels, got " + list.size()
            );
        }

        lastInput = list;
        preActivationOutputs = new ArrayList<>();
        List<double[][]> output = new ArrayList<>();

        // ✅ Кожен фільтр створює ОДНУ feature map
        for (int f = 0; f < filters.size(); f++) {
            output.add(convolveMultiChannel(list, filters.get(f), f));
        }

        return output;
    }

    /**
     * Виконує multi-channel згортку: один фільтр застосовується до всіх вхідних каналів.
     * Результати підсумовуються.
     */
    private double[][] convolveMultiChannel(List<double[][]> inputs, double[][][] filter, int filterIndex) {
        double[][][] paddedInputs = new double[inLength][][];
        for (int c = 0; c < inLength; c++) {
            paddedInputs[c] = applyPadding(inputs.get(c));
        }

        int paddedRows = paddedInputs[0].length;
        int paddedCols = paddedInputs[0][0].length;
        int fRows = filterSize;
        int fCols = filterSize;

        int outRows = (paddedRows - fRows) / stepSize + 1;
        int outCols = (paddedCols - fCols) / stepSize + 1;

        double[][] preActivationOutput = new double[outRows][outCols];
        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        for (int iStrides = 0; iStrides <= paddedRows - fRows; iStrides += stepSize) {
            int outCol = 0;
            for (int j = 0; j <= paddedCols - fCols; j += stepSize) {
                double sum = 0.0;

                // ✅ Підсумовуємо по всіх каналах
                for (int c = 0; c < inLength; c++) {
                    for (int x = 0; x < fRows; x++) {
                        for (int y = 0; y < fCols; y++) {
                            sum += paddedInputs[c][iStrides + x][j + y] * filter[c][x][y];
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
     * @param input   вхідна матриця розміром [rows][cols]
     * @return матриця розміром [rows + 2×padding][cols + 2×padding] з доданим padding
     */
    private double[][] applyPadding(double[][] input) {
        if (padding == 0) {
            return input;
        }

        int inRows = input.length;
        int inCols = input[0].length;
        int paddedRows = inRows + 2 * padding;
        int paddedCols = inCols + 2 * padding;

        double[][] padded = new double[paddedRows][paddedCols];

        // Копіюємо вхідні дані в центр padded матриці
        for (int i = 0; i < inRows; i++) {
            System.arraycopy(input[i], 0, padded[i + padding], padding, inCols);
        }

        return padded;
    }


    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        // dLdO.size() == filters.size() (по одному градієнту на фільтр)

        // КРОК 1: Gradient через activation
        List<double[][]> dLdZ = new ArrayList<>();
        for (int idx = 0; idx < dLdO.size(); idx++) {
            double[][] gradOutput = dLdO.get(idx);
            double[][] preActivation = preActivationOutputs.get(idx);
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
            filtersDelta.add(new double[inLength][filterSize][filterSize]);  // ✅ 3D
        }

        double[] biasesDelta = new double[filters.size()];

        // ✅ Градієнт по входу - окремо для кожного каналу
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();
        for (int c = 0; c < inLength; c++) {
            dLdOPreviousLayer.add(new double[inRows][inCols]);
        }

        // ✅ Проходимо по кожному фільтру
        for (int f = 0; f < filters.size(); f++) {
            double[][][] currFilter = filters.get(f);
            double[][] error = dLdZ.get(f);  // ✅ Один градієнт на фільтр

            double[][] spacedError = spaceArray(error);
            double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));

            // Градієнт по bias
            for (double[] row : error) {
                for (double val : row) {
                    biasesDelta[f] += val;
                }
            }

            // ✅ По кожному каналу окремо
            for (int c = 0; c < inLength; c++) {
                double[][] paddedInput = applyPadding(lastInput.get(c));

                // Градієнт по фільтру (канал c)
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

        // КРОК 3: Оновлення ваг
        for (int f = 0; f < filters.size(); f++) {
            for (int c = 0; c < inLength; c++) {
                // Добавляем L2 штраф к градиенту
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filtersDelta.get(f)[c][i][j] += l2Lamda * filters.get(f)[c][i][j];
                    }
                }
                multiply(filtersDelta.get(f)[c], -learningRate);
                add(filters.get(f)[c], filtersDelta.get(f)[c]);
            }
            biases[f] += -learningRate * biasesDelta[f];
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdOPreviousLayer);
        }
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

        if (stepSize == 1) {
            return input;
        }

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
     * Використовується під час backpropagation для обчислення градієнтів по фільтрах.
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


    @Override
    public int getOutputLength() {
        return filters.size();  // Кількість вихідних каналів = кількість фільтрів
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
            + filters.size();                                      // biases
    }

    @Override
    public String toString() {
        return String.format("🌀 CONVOLUTION | %d filters | %dx%d kernel | Stride: %d | Padding: %d | Parameters: %d",
            filters.size(), filterSize, filterSize, stepSize, padding, getParameterCount());
    }

}
