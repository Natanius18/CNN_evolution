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

    // Batch storage для backpropagation
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

    // ========== BATCH FORWARD PASS ==========

    @Override
    public List<double[]> getOutputBatch(List<List<double[][]>> batchInput) {
        // Конвертируем входы из feature maps в векторы
        List<double[]> batchVectors = new ArrayList<>();
        for (List<double[][]> input : batchInput) {
            batchVectors.add(matrixToVector(input));
        }

        // Forward pass для батча
        List<double[]> output = fullyConnectedForwardPassBatch(batchVectors);

        // Передаём следующему слою, если есть
        if (nextLayer != null) {
            // ❌ НЕПРАВИЛЬНО: nextLayer.getOutputBatch(null);
            // ✅ ПРАВИЛЬНО: передаём результат как батч List<List<double[][]>>

            // FC обычно последний слой перед выходом, но если есть ещё слои,
            // нужно конвертировать output обратно в List<List<double[][]>>
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
     * Forward pass для батча векторов
     * Каждый вектор в списке — это один пример из батча
     */
    public List<double[]> fullyConnectedForwardPassBatch(List<double[]> batchInputs) {
        List<double[]> batchOutputs = new ArrayList<>();
        lastXBatch = new ArrayList<>();
        lastZBatch = new ArrayList<>();

        for (double[] input : batchInputs) {
            // Валидация входа
            if (input.length != inLength) {
                throw new IllegalArgumentException(
                    "Expected input length " + inLength + ", got " + input.length
                );
            }

            // Forward для одного примера
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

            // Сохраняем для backprop
            lastXBatch.add(input.clone());
            lastZBatch.add(z.clone());

            // Применяем активацию
            double[] a = new double[outLength];
            for (int j = 0; j < outLength; j++) {
                a[j] = activation.forward(z[j]);
            }
            batchOutputs.add(a);
        }

        return batchOutputs;
    }

    // ========== BATCH BACKPROPAGATION ==========

    @Override
    public void backPropagationBatch(List<double[]> dLdOBatch) {
        int batchSize = dLdOBatch.size();

        // Инициализация аккумуляторов градиентов
        double[][] weightsDeltaSum = new double[inLength][outLength];
        double[] biasesDeltaSum = new double[outLength];

        List<double[]> dLdOPrevBatch = new ArrayList<>();

        // Обрабатываем каждый пример в батче
        for (int b = 0; b < batchSize; b++) {
            double[] dLda = dLdOBatch.get(b);  // градиент выхода
            double[] lastX = lastXBatch.get(b);  // вход
            double[] lastZ = lastZBatch.get(b);  // z перед активацией

            // === ЭТАП 1: Вычисление локальной ошибки ===
            // δ^(l) = ∂L/∂a^(l) ⊙ f'(z^(l))
            double[] delta = new double[outLength];
            for (int j = 0; j < outLength; j++) {
                delta[j] = dLda[j] * activation.backward(lastZ[j]);
            }

            // === ЭТАП 2: Вычисление градиента для предыдущего слоя ===
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

            // === ЭТАП 3: Аккумуляция градиентов параметров ===
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

            dLdOPrevBatch.add(dLdaPrev);
        }

        // === ЭТАП 4: Обновление параметров (усреднённые по батчу) ===
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                double grad = weightsDeltaSum[i][j] / batchSize;
                grad += l2Lambda * weights[i][j];  // L2 регуляризация
                weights[i][j] -= learningRate * grad;
            }
        }

        for (int j = 0; j < outLength; j++) {
            biases[j] -= learningRate * (biasesDeltaSum[j] / batchSize);
        }

        // Передаём батч градиентов предыдущему слою
        if (previousLayer != null) {
            previousLayer.backPropagationBatch(dLdOPrevBatch);
        }
    }

    // ========== WEIGHT INITIALIZATION ==========

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

    // ========== METADATA ==========

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
