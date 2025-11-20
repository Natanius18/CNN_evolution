package natanius.thesis.cnn.evolution.network;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.layers.ConvolutionLayer;
import natanius.thesis.cnn.evolution.layers.FullyConnectedLayer;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.layers.MaxPoolLayer;

public class NeuralNetwork {
    @Getter
    private final List<Layer> layers;

    private static final String RESET = "\u001B[0m";
    private static final String CYAN = "\u001B[36m";     // Titles
    private static final String GREEN = "\u001B[32m";    // Convolution
    private static final String BLUE = "\u001B[34m";     // Max Pooling
    private static final String MAGENTA = "\u001B[35m";  // Fully Connected
    private static final String YELLOW = "\u001B[33m";   // Stats

    public NeuralNetwork(List<Layer> layers) {
        this.layers = layers;
        linkLayers();
    }

    public void linkLayers() {
        if (layers.size() <= 1) {
            return;
        }

        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                layers.get(i).setNextLayer(layers.get(i + 1));
            } else if (i == layers.size() - 1) {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            } else {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
                layers.get(i).setNextLayer(layers.get(i + 1));
            }
        }
    }

    /**
     * Застосовує Softmax до вектора logits для отримання ймовірностей.
     *
     * @param logits вихідні значення з останнього шару
     * @return вектор ймовірностей (сума = 1.0)
     */
    private double[] applySoftmax(double[] logits) {
        // Для числової стабільності віднімаємо максимум
        double max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) max = logits[i];
        }

        double[] exp = new double[logits.length];
        double sum = 0.0;

        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < logits.length; i++) {
            exp[i] /= sum;
        }

        return exp;
    }

    /**
     * Обчислює градієнт Cross-Entropy Loss з Softmax.
     * Для Softmax + Cross-Entropy градієнт спрощується до: output - target
     *
     * @param networkOutput вихід мережі після Softmax (ймовірності)
     * @param correctAnswer правильна мітка класу (0-9)
     * @return градієнт loss function
     */
    private double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;  // One-hot encoding

        double[] errors = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            errors[i] = networkOutput[i] - expected[i];  // output - target
        }
        return errors;
    }


    private double computeCrossEntropyLoss(double[] output, int correctLabel) {
        double eps = 1e-7;
        return -Math.log(Math.max(output[correctLabel], eps));
    }


    private int getMaxIndex(double[] in) {
        double max = 0;
        int index = 0;

        for (int i = 0; i < in.length; i++) {
            if (in[i] >= max) {
                max = in[i];
                index = i;
            }
        }

        return index;
    }

    /**
     * Передбачення для одного зображення (одиночне)
     * Використовує batch size = 1 для inference
     */
    public int guess(Image image) {
        List<List<double[][]>> batchInputs = new ArrayList<>();
        List<double[][]> imgList = new ArrayList<>();
        imgList.add(image.data());
        batchInputs.add(imgList);

        // Forward через весь батч (розмір 1)
        List<double[]> batchOutputs = layers.getFirst().getOutputBatch(batchInputs);

        double[] output = batchOutputs.getFirst();
        double[] softmaxOut = applySoftmax(output);
        return getMaxIndex(softmaxOut);
    }

    /**
     * Передбачення для батчу зображень
     */
    public List<Integer> guessBatch(List<Image> images) {
        List<Integer> predictions = new ArrayList<>();

        List<List<double[][]>> batchInputs = new ArrayList<>();
        for (Image img : images) {
            List<double[][]> imgList = new ArrayList<>();
            imgList.add(img.data());
            batchInputs.add(imgList);
        }

        List<double[]> batchOutputs = layers.getFirst().getOutputBatch(batchInputs);

        for (double[] output : batchOutputs) {
            double[] softmaxOut = applySoftmax(output);
            predictions.add(getMaxIndex(softmaxOut));
        }

        return predictions;
    }

    /**
     * Real-time prediction для одного вектора (784 елементів для MNIST)
     */
    public double[] guessInRealTime(double[] inputs) {
        double[][] inputMatrix = new double[28][28];

        for (int i = 0; i < 28; i++) {
            System.arraycopy(inputs, i * 28, inputMatrix[i], 0, 28);
        }

        List<List<double[][]>> batchInputs = new ArrayList<>();
        List<double[][]> imgList = new ArrayList<>();
        imgList.add(inputMatrix);
        batchInputs.add(imgList);

        List<double[]> batchOutputs = layers.getFirst().getOutputBatch(batchInputs);

        double[] output = batchOutputs.getFirst();
        return applySoftmax(output);  // Повертаємо ймовірності
    }



    public float test(List<Image> images) {
        int correct = 0;
        int size = images.size();

        for (Image img : images) {
            int guess = guess(img);
            if (guess == img.label()) {
                correct++;
            }
        }

        return ((float) correct / size);
    }

    /**
     * Тестування на батчах (більш ефективно для великих наборів)
     */
    public float testBatch(List<Image> images, int batchSize) {
        int correct = 0;
        int numBatches = (images.size() + batchSize - 1) / batchSize;

        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, images.size());
            List<Image> batch = images.subList(start, end);

            List<Integer> predictions = guessBatch(batch);

            for (int i = 0; i < predictions.size(); i++) {
                if (predictions.get(i) == batch.get(i).label()) {
                    correct++;
                }
            }
        }

        return ((float) correct / images.size());
    }

    /**
     * Навчання на одній епосі з mini-batch розбиттям
     *
     * @param images    тренувальний набір
     * @param batchSize розмір батча
     */
    public void trainEpoch(List<Image> images, int batchSize) {
        int numBatches = (images.size() + batchSize - 1) / batchSize;
        double totalLoss = 0.0;

        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, images.size());
            List<Image> batch = images.subList(start, end);

            List<List<double[][]>> batchInputs = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            for (Image img : batch) {
                List<double[][]> imgList = new ArrayList<>();
                imgList.add(img.data());
                batchInputs.add(imgList);
                labels.add(img.label());
            }

            // Forward через всю мережу
            List<double[]> batchOutputs = layers.getFirst().getOutputBatch(batchInputs);

            List<double[]> batchErrors = new ArrayList<>();
            double batchLoss = 0.0;

            for (int i = 0; i < batch.size(); i++) {
                double[] output = batchOutputs.get(i);
                double[] softmaxOut = applySoftmax(output);

                // Обчислюємо loss для цього прикладу
                double loss = computeCrossEntropyLoss(softmaxOut, labels.get(i));
                batchLoss += loss;

                // Обчислюємо градієнт (Softmax + CrossEntropy)
                double[] errors = getErrors(softmaxOut, labels.get(i));
                batchErrors.add(errors);
            }

            totalLoss += batchLoss;

            layers.getLast().backPropagationBatch(batchErrors);
        }
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(CYAN).append("\n╔════════════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append("║ ").append(centerText("🧠 NEURAL NETWORK ARCHITECTURE 🧠", 81)).append(" ║\n");
        sb.append("╠════════════════════════════════════════════════════════════════════════════════════╣\n").append(RESET);

        int totalParams = 0;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            totalParams += layer.getParameterCount();

            String color;
            if (layer instanceof ConvolutionLayer) {
                color = GREEN;
            } else if (layer instanceof MaxPoolLayer) {
                color = BLUE;
            } else {
                color = (layer instanceof FullyConnectedLayer) ? MAGENTA : RESET;
            }

            sb.append(color).append("║ ").append(centerText(layer.toString(), 82)).append(" ║\n").append(RESET);

            if (i < layers.size() - 1) {
                sb.append("║                                        ▼                                           ║\n");
            }
        }

        sb.append("╚════════════════════════════════════════════════════════════════════════════════════╝\n");
        sb.append(YELLOW).append("📊 Total Layers: ").append(layers.size())
            .append(" | Total Parameters: ").append(totalParams).append(RESET).append("\n");

        return sb.toString();
    }

    private String centerText(String text, int width) {
        int padding = Math.max(0, (width - text.length()));
        int leftPadding = padding / 2;
        int rightPadding = padding - leftPadding;

        return " ".repeat(leftPadding) + text + " ".repeat(rightPadding);
    }
}
