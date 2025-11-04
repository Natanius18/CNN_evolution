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
    private static final String RED = "\u001B[31m";
    private static final String CYAN = "\u001B[36m";  // Titles
    private static final String GREEN = "\u001B[32m"; // Convolution
    private static final String BLUE = "\u001B[34m";  // Max Pooling
    private static final String MAGENTA = "\u001B[35m"; // Fully Connected
    private static final String YELLOW = "\u001B[33m"; // Stats


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
    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;
        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;  // One-hot encoding

        double[] errors = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            errors[i] = networkOutput[i] - expected[i];  // output - target
        }
        return errors;
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

    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(image.data());
        double[] out = layers.getFirst().getOutput(inList);
        double[] softmaxOut = applySoftmax(out);
        return getMaxIndex(softmaxOut);
    }

    public float test(List<Image> images) {
        int correct = 0;

        int size = images.size();
        for (int i = 0; i < size; i++) {
//            printProgress(i, size, "Testing         ");
            Image img = images.get(i);
            int guess = guess(img);

            if (guess == img.label()) {
                correct++;
            }
        }
//        if (DEBUG) {
//            System.out.println();
//        }
        return ((float) correct / size);
    }


    public void train(List<Image> images) {
        int size = images.size();

        for (int i = 0; i < size; i++) {
//            printProgress(i, size, "Training        ");

            Image img = images.get(i);
            List<double[][]> inList = new ArrayList<>();
            inList.add(img.data());

            // Forward pass
            double[] out = layers.getFirst().getOutput(inList);

            // Застосовуємо Softmax для отримання ймовірностей
            double[] softmaxOut = applySoftmax(out);

            // Обчислюємо градієнт Cross-Entropy Loss
            double[] dldO = getErrors(softmaxOut, img.label());

            // Backpropagation
            layers.getLast().backPropagation(dldO);
        }

//        if (DEBUG) {
//            System.out.println();
//        }
    }

//    private static void printProgress(int i, int totalImages, String processName) {
//        if (DEBUG) {
//            double progress = (i + 1) * 100. / totalImages;
//            String progressBar = "[" + "■".repeat((int) (progress / 2)) + " ".repeat((int) (50 - progress / 2)) + "]";
//
//            String progressBarColor;
//            if (progress < 50) {
//                progressBarColor = RED;
//            } else {
//                progressBarColor = progress < 80 ? YELLOW : GREEN;
//            }
//
//            String formattedProgress = String.format("%.2f", progress);
//            System.out.print("\r" + processName + " progress: \u001B[1m" + progressBarColor + progressBar + RESET + "\u001B[1m " + formattedProgress + "%" + RESET);
//        }
//    }

    public double[] guessInRealTime(double[] inputs) {
        double[][] inputMatrix = new double[28][28];

        for (int i = 0; i < 28; i++) {
            System.arraycopy(inputs, i * 28, inputMatrix[i], 0, 28);
        }
        List<double[][]> inList = new ArrayList<>();
        inList.add(inputMatrix);

        double[] out = layers.getFirst().getOutput(inList);
        return applySoftmax(out);  // Повертаємо ймовірності
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(CYAN).append("\n╔════════════════════════════════════════════════════════════════════════╗\n");
        sb.append("║ ").append(centerText("🧠 NEURAL NETWORK ARCHITECTURE 🧠", 70)).append(" ║\n");
        sb.append("╠════════════════════════════════════════════════════════════════════════╣\n").append(RESET);

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

            sb.append(color).append("║ ").append(centerText(layer.toString(), 70)).append(" ║\n").append(RESET);

            if (i < layers.size() - 1) {
                sb.append("║                                  ▼                                     ║\n");
            }
        }

        sb.append("╚════════════════════════════════════════════════════════════════════════╝\n");
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
