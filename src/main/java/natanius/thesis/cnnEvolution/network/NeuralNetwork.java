package natanius.thesis.cnnEvolution.network;

import static natanius.thesis.cnnEvolution.data.MatrixUtility.add;
import static natanius.thesis.cnnEvolution.data.MatrixUtility.multiply;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnnEvolution.data.Image;
import natanius.thesis.cnnEvolution.layers.ConvolutionLayer;
import natanius.thesis.cnnEvolution.layers.FullyConnectedLayer;
import natanius.thesis.cnnEvolution.layers.Layer;
import natanius.thesis.cnnEvolution.layers.MaxPoolLayer;

public class NeuralNetwork {
    private final List<Layer> layers;
    private final int scaleFactor;

    public static final String RESET = "\u001B[0m";
    public static final String CYAN = "\u001B[36m";  // Titles
    public static final String GREEN = "\u001B[32m"; // Convolution
    public static final String BLUE = "\u001B[34m";  // Max Pooling
    public static final String MAGENTA = "\u001B[35m"; // Fully Connected
    public static final String YELLOW = "\u001B[33m"; // Stats


    public NeuralNetwork(List<Layer> layers, int scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers() {

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

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
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
        inList.add(multiply(image.data(), (1.0 / scaleFactor)));

        double[] out = layers.getFirst().getOutput(inList);
        return getMaxIndex(out);
    }

    public float test(List<Image> images) {
        int correct = 0;

        for (Image img : images) {
            int guess = guess(img);

            if (guess == img.label()) {
                correct++;
            }
        }

        return ((float) correct / images.size());
    }

    public void train(List<Image> images) {
        int totalImages = images.size();

        for (int i = 0; i < totalImages; i++) {
            printProgress(i, totalImages);

            Image img = images.get(i);
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.data(), (1.0 / scaleFactor)));

            double[] out = layers.getFirst().getOutput(inList);
            double[] dldO = getErrors(out, img.label());

            layers.getLast().backPropagation(dldO);
        }
        System.out.println();
    }

    private static void printProgress(int i, int totalImages) {
        double progress = (i + 1) * 100. / totalImages;
        String progressBar = "[" + "■".repeat((int) (progress / 2)) + " ".repeat((int) (50 - progress / 2)) + "]";

        String progressBarColor = progress < 50 ? "\u001B[31m" : (progress < 80 ? "\u001B[33m" : "\u001B[32m");

        String formattedProgress = String.format("%.2f", progress);
        System.out.print("\r" + "Training progress: \u001B[1m" + progressBarColor + progressBar + "\u001B[0m\u001B[1m " + formattedProgress + "%\u001B[0m");
    }

    public double[] guessInRealTime(double[] inputs) {
        double[][] inputMatrix = new double[28][28];

        for (int i = 0; i < 28; i++) {
            System.arraycopy(inputs, i * 28, inputMatrix[i], 0, 28);
        }
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(inputMatrix, (1.0 / 255)));
        return layers.getFirst().getOutput(inList);
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

            String color = (layer instanceof ConvolutionLayer) ? GREEN :
                (layer instanceof MaxPoolLayer) ? BLUE :
                    (layer instanceof FullyConnectedLayer) ? MAGENTA : RESET;

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
