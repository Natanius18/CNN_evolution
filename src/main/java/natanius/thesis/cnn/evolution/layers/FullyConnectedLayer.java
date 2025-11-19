package natanius.thesis.cnn.evolution.layers;

import static natanius.thesis.cnn.evolution.data.Constants.L2_REGULARIZATION_LAMBDA;
import static natanius.thesis.cnn.evolution.data.Constants.OUTPUT_CLASSES;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.activation.LeakyReLU;
import natanius.thesis.cnn.evolution.activation.Linear;
import natanius.thesis.cnn.evolution.activation.ReLU;
import natanius.thesis.cnn.evolution.activation.Sigmoid;

public class FullyConnectedLayer extends Layer {

    private final Activation activation;
    private final double[][] weights;
    private final double[] biases;
    private final int inLength;
    private final int outLength;
    private final double learningRate;
    private double l2Lambda = L2_REGULARIZATION_LAMBDA;
    private List<double[]> lastXBatch;
    private List<double[]> lastZBatch;

    public FullyConnectedLayer(Activation activation, int inLength, double learningRate) {
        this(activation, inLength, OUTPUT_CLASSES, learningRate);
    }

    public FullyConnectedLayer(Activation activation, int inLength, int outLength, double learningRate) {
        this.activation = activation;
        this.inLength = inLength;
        this.outLength = outLength;
        this.learningRate = learningRate;

        weights = new double[inLength][outLength];
        if (activation instanceof ReLU || activation instanceof LeakyReLU || activation instanceof Linear) {
            initWeightsHe();
        } else if (activation instanceof Sigmoid) {
            initWeightsXavier();
        } else {
            throw new IllegalArgumentException(
                "Unsupported activation function: " + activation.getClass().getSimpleName() +
                    ". Supported: ReLU, LeakyReLU, Sigmoid"
            );
        }

        biases = new double[outLength];
    }


    @Override
    public List<double[]> getOutputBatch(List<List<double[][]>> batchInput) {
        // Конвертуємо входи з feature maps у вектори
        List<double[]> batchVectors = new ArrayList<>();
        for (List<double[][]> input : batchInput) {
            batchVectors.add(matrixToVector(input));
        }

        // Forward pass для батча
        List<double[]> output = fullyConnectedForwardPassBatch(batchVectors);

        // Передаємо наступному шару, якщо є
        if (nextLayer != null) {
            List<List<double[][]>> outputAsFeatureMaps = new ArrayList<>();
            for (double[] vec : output) {
                List<double[][]> featureMap = vectorToMatrix(vec, 1, 1, vec.length);
                outputAsFeatureMaps.add(featureMap);
            }
            return nextLayer.getOutputBatch(outputAsFeatureMaps);
        }
        return output;
    }

    /**
     * Реалізує прямий прохід (forward propagation) через повнозв'язний шар.
     *
     * <p><b>Математичні позначення:</b>
     * <ul>
     *   <li>input = a^(l-1) — активації попереднього шару (вхід поточного шару)</li>
     *   <li>weights = W^(l) — матриця ваг поточного шару</li>
     *   <li>biases = b^(l) — вектор зміщень поточного шару</li>
     *   <li>z = z^(l) — зважена сума (перед активацією)</li>
     *   <li>a = a^(l) — активації після застосування функції активації</li>
     * </ul>
     *
     * <p><b>Крок 1:</b> Обчислення зваженої суми:
     * <pre>
     *   z^(l) = W^(l) · a^(l-1) + b^(l)
     * </pre>
     * де · означає матричне множення (у коді: weights[i][j] * input[i]).
     *
     * <p><b>Крок 2:</b> Застосування функції активації:
     * <pre>
     *   a^(l) = f^(l)(z^(l))
     * </pre>
     * де f^(l) — функція активації шару (ReLU, Sigmoid тощо).
     *
     * <p><b>Оптимізація:</b> Під час обчислення z пропускаються нульові елементи input[i],
     * що особливо ефективно після ReLU активації або pooling операцій.
     *
     * <p><b>Збереження для backpropagation:</b>
     * <ul>
     *   <li>lastX = a^(l-1) — вхідні активації</li>
     *   <li>lastZ = z^(l) — зважена сума перед активацією</li>
     * </ul>
     *
     * @param batchInputs список векторів a^(l-1) — активації попереднього шару для кожного прикладу в батчі
     * @return список векторів a^(l) — активації поточного шару після застосування f^(l)
     */
    public List<double[]> fullyConnectedForwardPassBatch(List<double[]> batchInputs) {
        List<double[]> batchOutputs = new ArrayList<>();
        lastXBatch = new ArrayList<>();
        lastZBatch = new ArrayList<>();

        for (double[] input : batchInputs) {
            validateInput(input);

            double[] z = biases.clone();

            for (int i = 0; i < inLength; i++) {
                double aPrevI = input[i];
                if (aPrevI != 0.0) {
                    double[] wRow = weights[i];
                    for (int j = 0; j < outLength; j++) {
                        z[j] += wRow[j] * aPrevI;
                    }
                }
            }

            // Збереження a^(l-1) для backpropagation
            lastXBatch.add(input.clone());
            lastZBatch.add(z.clone());

            double[] a = applyActivation(z);
            batchOutputs.add(a);
        }

        return batchOutputs;
    }

    private void validateInput(double[] input) {
        if (input.length != inLength) {
            throw new IllegalArgumentException(
                "Expected input length " + inLength + ", got " + input.length
            );
        }
    }

    /**
     * Застосовує функцію активації f^(l) до кожного елемента вектора z^(l).
     *
     * @param z вектор z^(l) — зважена сума
     * @return вектор a^(l) = f^(l)(z^(l)) — активації після застосування функції
     */
    private double[] applyActivation(double[] z) {
        double[] a = new double[outLength];
        for (int j = 0; j < outLength; j++) {
            a[j] = activation.forward(z[j]);
        }
        return a;
    }


    /**
     * Реалізує алгоритм зворотного поширення помилки (backpropagation) через повнозв'язний шар.
     * Процес складається з чотирьох етапів: обчислення локальної похибки, обчислення градієнта
     * для попереднього шару, оновлення ваг та оновлення зміщень.
     *
     * <p><b>Математичні позначення:</b>
     * <ul>
     *   <li>dLda = ∂L/∂a^(l) — градієнт втрат відносно виходу шару (вхідний параметр)</li>
     *   <li>delta = δ^(l) — локальна похибка шару</li>
     *   <li>dLdX = ∂L/∂a^(l-1) — градієнт втрат відносно входу шару</li>
     *   <li>lastZ = z^(l) — зважена сума перед активацією (збережена з forward pass)</li>
     *   <li>lastX = a^(l-1) — активації попереднього шару (збережені з forward pass)</li>
     *   <li>weights = W^(l) — матриця ваг</li>
     *   <li>biases = b^(l) — вектор зміщень</li>
     * </ul>
     *
     * <p><b>ЕТАП 1: Обчислення локальної похибки</b>
     * <pre>
     *   δ^(l) = ∂L/∂a^(l) ⊙ f'(z^(l))
     * </pre>
     * де ⊙ — поелементне множення (Hadamard product), f' — похідна функції активації.
     * <p>
     *
     * <p><b>ЕТАП 2: Обчислення градієнта для попереднього шару</b>
     * <pre>
     *   ∂L/∂a^(l-1) = (W^(l))^T · δ^(l)
     * </pre>
     * Цей градієнт передається попередньому шару для продовження backpropagation.
     * <p>
     *
     * <p><b>ЕТАП 3: Обчислення градієнтів параметрів</b>
     * <pre>
     *   ∂L/∂W^(l)_ij = a^(l-1)_i · δ^(l)_j
     *   ∂L/∂b^(l)_j = δ^(l)_j
     * </pre>
     * <p>
     *
     * <p><b>ЕТАП 4: Оновлення параметрів методом градієнтного спуску</b>
     * <pre>
     *   W^(l) := W^(l) - η · ∂L/∂W^(l)
     *   b^(l) := b^(l) - η · ∂L/∂b^(l)
     * </pre>
     * де η — швидкість навчання (learning rate).
     *
     * <p><b>ВАЖЛИВО:</b> Градієнт dLdX обчислюється ДО оновлення ваг, використовуючи
     * старі значення параметрів. Це критично для коректності backpropagation через весь ланцюг шарів.
     *
     * @param dLdaBatch градієнт функції втрат відносно виходу шару (∂L/∂a^(l))
     */
    @Override
    public void backPropagationBatch(List<double[]> dLdaBatch) {
        int batchSize = dLdaBatch.size();

        // Ініціалізація акумуляторів градієнтів
        double[][] weightsDeltaSum = new double[inLength][outLength];
        double[] biasesDeltaSum = new double[outLength];

        List<double[]> dLdaPrevBatch = new ArrayList<>();

        // Обробляємо кожен приклад у батчі
        for (int b = 0; b < batchSize; b++) {
            double[] dLda = dLdaBatch.get(b);  // градієнт виходу
            double[] lastX = lastXBatch.get(b);  // вхід
            double[] lastZ = lastZBatch.get(b);  // z перед активацією

            // ЕТАП 1: Обчислення локальної похибки 
            // δ^(l) = ∂L/∂a^(l) ⊙ f'(z^(l))
            double[] delta = new double[outLength];
            for (int j = 0; j < outLength; j++) {
                delta[j] = dLda[j] * activation.backward(lastZ[j]);
            }

            // ЕТАП 2: Обчислення градієнта для попереднього шару 
            // ∂L/∂a^(l-1) = (W^(l))^T · δ^(l)
            double[] dLdaPrev = new double[inLength];
            for (int i = 0; i < inLength; i++) {
                double sum = 0.0;
                double[] wRow = weights[i];
                for (int j = 0; j < outLength; j++) {
                    sum += wRow[j] * delta[j];
                }
                dLdaPrev[i] = sum;
            }

            // ЕТАП 3: Акумуляція градієнтів параметрів 
            // ∂L/∂W^(l)_ij = a^(l-1)_i · δ^(l)_j
            for (int i = 0; i < inLength; i++) {
                double aPrevI = lastX[i];
                for (int j = 0; j < outLength; j++) {
                    double dLdWij = aPrevI * delta[j];
                    weightsDeltaSum[i][j] += dLdWij;
                }
            }

            // ∂L/∂b^(l)_j = δ^(l)_j
            for (int j = 0; j < outLength; j++) {
                biasesDeltaSum[j] += delta[j];
            }

            dLdaPrevBatch.add(dLdaPrev);
        }

        // ЕТАП 4: Оновлення параметрів (усереднені по батчу) 
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                double grad = weightsDeltaSum[i][j] / batchSize;
                grad += l2Lambda * weights[i][j];  // L2 регуляризація
                weights[i][j] -= learningRate * grad;
            }
        }

        for (int j = 0; j < outLength; j++) {
            biases[j] -= learningRate * (biasesDeltaSum[j] / batchSize);
        }

        if (previousLayer != null) {
            previousLayer.backPropagationBatch(dLdaPrevBatch);
        }
    }


    private void initWeightsHe() {
        double std = Math.sqrt(2.0 / inLength);
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                weights[i][j] = RANDOM.nextGaussian() * std;
            }
        }
    }

    private void initWeightsXavier() {
        double limit = Math.sqrt(6.0 / (inLength + outLength));
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                weights[i][j] = (RANDOM.nextDouble() * 2 - 1) * limit;
            }
        }
    }


    @Override
    public int getOutputLength() {
        return outLength;
    }

    @Override
    public int getOutputRows() {
        return 1;
    }

    @Override
    public int getOutputCols() {
        return outLength;
    }

    @Override
    public int getOutputElements() {
        return outLength;
    }

    @Override
    public int getParameterCount() {
        return inLength * outLength + outLength;
    }

    @Override
    public String toString() {
        return String.format("🔗 FULLY CONNECTED | Inputs: %d → Outputs: %d | Parameters: %d",
            inLength, outLength, getParameterCount());
    }
}
