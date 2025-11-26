package natanius.thesis.cnn.evolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.DATASET_FRACTION;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTestData;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTrainData;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.genes.Chromosome;
import natanius.thesis.cnn.evolution.genes.LayerGene;
import natanius.thesis.cnn.evolution.genes.LayerType;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class BestArchitectureDistr {


    public static void main(String[] args) {
        List<Image> imagesTrain = loadTrainData();
        List<Image> imagesTest = loadTestData();
        imagesTrain = imagesTrain.subList(0, (int) (imagesTrain.size() * DATASET_FRACTION));
        System.out.println("Sizes: " + imagesTrain.size() + " " + imagesTest.size());

        String text = "CONVOLUTION (64 filters 7x7, stride=2, same padding + ReLU) → FC output";

        Chromosome chromosome = parseChromosomeString(text);
        NeuralNetwork network = buildNetworkFromChromosome(chromosome);

        System.out.println(network);

        for (int epoch = 1; epoch <= 3; epoch++) {
            long start = now().getEpochSecond();
            shuffle(imagesTrain, RANDOM);
            double loss = network.trainEpoch(imagesTrain, 32);
            float testAccuracy = network.test(imagesTest);
            float trainAccuracy = network.test(imagesTrain);
            long trainingTime = now().getEpochSecond() - start;
            System.out.printf("%s%nEpoch %d: Loss = %.5f, Train Accuracy = %.5f, Test Accuracy = %.5f%%%n", text, epoch, loss, trainAccuracy, testAccuracy);
            printTimeTaken(trainingTime);
        }
        analyzeClassDistribution(network, imagesTest);
    }


    public static Chromosome parseChromosomeString(String str) {
        List<LayerGene> genes = new ArrayList<>();

        String[] parts = str.split("→");
        for (String raw : parts) {
            String part = raw.trim();

            if (part.startsWith("CONVOLUTION")) {
                String inside = part.substring(part.indexOf('(') + 1, part.lastIndexOf(')'));

                int filters = Integer.parseInt(inside.split("filters")[0].trim());
                String afterFilters = inside.split("filters")[1].trim();

                String filterSizeStr = afterFilters.split(",")[0].trim();     // "5x5"
                int filterSize = Integer.parseInt(filterSizeStr.split("x")[0]);

                String strideStr = inside.replaceAll(".*stride=", "").split("[,)]")[0];
                int stride = Integer.parseInt(strideStr.trim());

                String paddingType = inside.contains("same") ? "same" : "valid";
                int padding = paddingType.equals("same") ? filterSize / 2 : 0;

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

            else if (part.startsWith("MAX_POOL")) {
                String inside = part.substring(part.indexOf('(') + 1, part.lastIndexOf(')'));

                String winStr = inside.split(",")[0].trim();   // "3x3"
                int window = Integer.parseInt(winStr.split("x")[0]);

                String strideStr = inside.replaceAll(".*stride=", "").trim();
                int stride = Integer.parseInt(strideStr);

                genes.add(new LayerGene(LayerType.MAX_POOL, window, stride));
            }

            else if (part.startsWith("FC")) {
                if (part.contains("output")) {
                    genes.add(new LayerGene(LayerType.FULLY_CONNECTED));
                } else {
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

    /**
     * Анализирует распределение угадываний модели по классам
     *
     * @param network обученная нейросеть
     * @param images  тестовый набор данных
     */
    public static void analyzeClassDistribution(NeuralNetwork network, List<Image> images) {
        int[][] distribution = new int[10][2]; // [класс][угаданные, всего]
        
        for (Image img : images) {
            int label = img.label();
            int prediction = network.guess(img);
            
            distribution[label][1]++; // всего картинок этого класса
            if (prediction == label) {
                distribution[label][0]++; // угаданные
            }
        }
        
        // Вывод результатов
        System.out.println("\n=== Class Distribution ===");
        for (int i = 0; i < 10; i++) {
            int correct = distribution[i][0];
            int total = distribution[i][1];
            float accuracy = total > 0 ? (float) correct / total * 100 : 0;
            System.out.printf("Class %d: %d/%d (%.2f%%)%n", i, correct, total, accuracy);
        }

    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n%n", minutes, seconds);
    }

}

