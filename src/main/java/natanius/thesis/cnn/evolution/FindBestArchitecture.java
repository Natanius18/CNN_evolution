package natanius.thesis.cnn.evolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTestData;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTrainData;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.genes.Chromosome;
import natanius.thesis.cnn.evolution.genes.LayerGene;
import natanius.thesis.cnn.evolution.genes.LayerType;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class FindBestArchitecture {


    public static void main(String[] args) {
        List<Image> imagesTrain = loadTrainData();
        List<Image> imagesTest = loadTestData();
        System.out.println("Sizes: " + imagesTrain.size() + " " + imagesTest.size());

        String[] chromosomes = {
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (64 filters 5x5, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (8 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → CONVOLUTION (32 filters 7x7, stride=2, valid padding + ReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (8 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → FC output",
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → CONVOLUTION (32 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (16 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → CONVOLUTION (32 filters 7x7, stride=2, valid padding + ReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (16 filters 7x7, stride=1, same padding + LeakyReLU) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (64 filters 3x3, stride=1, valid padding + ReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (64 filters 5x5, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → FC output",
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → FC output",
            "CONVOLUTION (16 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (64 filters 7x7, stride=2, same padding + ReLU) → FC output",
            "CONVOLUTION (16 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (4 filters 5x5, stride=1, same padding + ReLU) → MAX_POOL (2x2, stride=1) → CONVOLUTION (32 filters 7x7, stride=2, valid padding + ReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (64 filters 3x3, stride=1, same padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → CONVOLUTION (64 filters 5x5, stride=1, same padding + ReLU) → FC output",
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → MAX_POOL (2x2, stride=1) → CONVOLUTION (32 filters 7x7, stride=2, valid padding + ReLU) → MAX_POOL (2x2, stride=1) → FC output",
            "CONVOLUTION (8 filters 7x7, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (32 filters 5x5, stride=1, valid padding + LeakyReLU) → CONVOLUTION (64 filters 3x3, stride=1, valid padding + LeakyReLU) → MAX_POOL (3x3, stride=2) → FC output",
            "CONVOLUTION (16 filters 3x3, stride=1, same padding + LeakyReLU) → MAX_POOL (3x3, stride=1) → FC output"
        };

        Arrays.stream(chromosomes)
            .parallel()
            .forEach(text -> testOneNetwork(imagesTrain, imagesTest, text));
    }


    private static void testOneNetwork(List<Image> imagesTrain, List<Image> imagesTest, String text) {

        Chromosome chromosome = parseChromosomeString(text);
        NeuralNetwork network = buildNetworkFromChromosome(chromosome);

        System.out.println(network);

        for (int epoch = 1; epoch <= 10; epoch++) {
            long start = now().getEpochSecond();
            shuffle(imagesTrain, RANDOM);
            network.trainEpoch(imagesTrain, 32);
            float testAccuracy = network.test(imagesTest);
            float trainAccuracy = network.test(imagesTrain);
            System.out.printf("%s%nEpoch %d: Train Accuracy = %.2f, Test Accuracy = %.2f%%%n", text, epoch, trainAccuracy, testAccuracy);
            printTimeTaken(now().getEpochSecond() - start);
        }

    }

    public static Chromosome parseChromosomeString(String str) {
        List<LayerGene> genes = new ArrayList<>();

        String[] parts = str.split("→");
        for (String raw : parts) {
            String part = raw.trim();

            // ---------- CONVOLUTION ----------
            if (part.startsWith("CONVOLUTION")) {
                // Example: "CONVOLUTION (4 filters 5x5, stride=1, same padding + Sigmoid)"
                String inside = part.substring(part.indexOf('(') + 1, part.lastIndexOf(')'));

                // filters & filter size
                int filters = Integer.parseInt(inside.split("filters")[0].trim());
                String afterFilters = inside.split("filters")[1].trim();

                String filterSizeStr = afterFilters.split(",")[0].trim();     // "5x5"
                int filterSize = Integer.parseInt(filterSizeStr.split("x")[0]);

                // stride
                String strideStr = inside.replaceAll(".*stride=", "").split("[,)]")[0];
                int stride = Integer.parseInt(strideStr.trim());

                // padding
                String paddingType = inside.contains("same") ? "same" : "valid";
                int padding = paddingType.equals("same") ? filterSize / 2 : 0;

                // activation
                String act = inside.substring(inside.indexOf('+') + 1).trim();
                Activation activation = ACTIVATION_STRATEGIES[
                    findActivationIndex(act)
                    ];

                genes.add(new LayerGene(
                    LayerType.CONVOLUTION,
                    filters,
                    filterSize,
                    activation,
                    padding,
                    stride
                ));
            }

            // ---------- MAX POOL ----------
            else if (part.startsWith("MAX_POOL")) {
                // Example: "MAX_POOL (3x3, stride=2)"
                String inside = part.substring(part.indexOf('(') + 1, part.lastIndexOf(')'));

                String winStr = inside.split(",")[0].trim();   // "3x3"
                int window = Integer.parseInt(winStr.split("x")[0]);

                String strideStr = inside.replaceAll(".*stride=", "").trim();
                int stride = Integer.parseInt(strideStr);

                genes.add(new LayerGene(LayerType.MAX_POOL, window, stride));
            }

            // ---------- FC ----------
            else if (part.startsWith("FC")) {
                if (part.contains("output")) {
                    genes.add(new LayerGene(LayerType.FULLY_CONNECTED));
                } else {
                    // Example: "FC (128 + ReLU)"
                    String inside = part.substring(part.indexOf('(') + 1, part.lastIndexOf(')'));
                    int size = Integer.parseInt(inside.split("\\+")[0].trim());

                    String actStr = inside.substring(inside.indexOf('+') + 1).trim();
                    Activation activation = ACTIVATION_STRATEGIES[
                        findActivationIndex(actStr)
                        ];

                    genes.add(new LayerGene(LayerType.FULLY_CONNECTED, size, activation));
                }
            }
        }

        return new Chromosome(genes);
    }

    private static int findActivationIndex(String name) {
        for (int i = 0; i < ACTIVATION_STRATEGIES.length; i++) {
            if (ACTIVATION_STRATEGIES[i].getClass().getName().split("\\.")[5].equalsIgnoreCase(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Unknown activation: " + name);
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n%n", minutes, seconds);
    }

}

