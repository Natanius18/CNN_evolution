package natanius.thesis.cnn.evolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static natanius.thesis.cnn.evolution.data.Constants.BATCH_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.DATASET_FRACTION;
import static natanius.thesis.cnn.evolution.data.Constants.EPOCHS;
import static natanius.thesis.cnn.evolution.data.Constants.GENERATIONS;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTestData;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTrainData;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;
import static natanius.thesis.cnn.evolution.genes.PopulationGenerator.generateInitialPopulation;

import java.util.Comparator;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.LeakyReLU;
import natanius.thesis.cnn.evolution.activation.Linear;
import natanius.thesis.cnn.evolution.data.ExcelLogger;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.genes.GeneticAlgorithm;
import natanius.thesis.cnn.evolution.genes.Individual;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.network.EpochTrainer;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;
import natanius.thesis.cnn.evolution.visualization.FormDigits;

public class Evolution {

    private static final int MODE = 1;
    private static final GeneticAlgorithm GA = new GeneticAlgorithm();
    private static final EpochTrainer EPOCH_TRAINER = new EpochTrainer(BATCH_SIZE, EPOCHS);

    public static void main(String[] args) {

        List<Image> imagesTrain = loadTrainData();
        List<Image> imagesTest = loadTestData();
        imagesTrain = imagesTrain.subList(0, (int) (imagesTrain.size() * DATASET_FRACTION));
        imagesTest = imagesTest.subList(0, (int) (imagesTest.size() * DATASET_FRACTION));
        System.out.println("Sizes: " + imagesTrain.size() + " " + imagesTest.size());

        if (MODE == 1) {
            testOneNetwork(imagesTrain, imagesTest);
        } else if (MODE == 2) {
            runGeneticAlgorithm(imagesTrain, imagesTest);
        }
    }

    private static void testOneNetwork(List<Image> imagesTrain, List<Image> imagesTest) {

        NeuralNetwork network = new NetworkBuilder()
            .addConvolutionLayer(8, 3, 1, 0.001, new LeakyReLU(), 0)
            .addMaxPoolLayer(2, 1)
            .addFullyConnectedLayer(0.001, new Linear())
            .build();

//        EpochTrainer epochTrainer = new EpochTrainer();
//        epochTrainer.train(network, imagesTrain, imagesTest);

        System.out.println(network);

        for (int epoch = 1; epoch <= 10; epoch++) {
            long start = now().getEpochSecond();
            network.trainEpoch(imagesTrain, BATCH_SIZE);
            float testAccuracy = network.test(imagesTest);
            float trainAccuracy = network.test(imagesTrain);
            System.out.printf("Epoch %d: Train Accuracy = %.2f, Test Accuracy = %.2f%%%n", epoch, trainAccuracy, testAccuracy);
            printTimeTaken(now().getEpochSecond() - start);
        }

        new Thread(new FormDigits(network)).start();
    }

    private static void runGeneticAlgorithm(List<Image> imagesTrain, List<Image> imagesTest) {

        List<Individual> population = generateInitialPopulation();

        List<Image> validationSet = imagesTrain.subList(0, imagesTrain.size() / 10);
        List<Image> trainSet = imagesTrain.subList(imagesTrain.size() / 10, imagesTrain.size());
        for (int gen = 0; gen < GENERATIONS; gen++) {
            long start = now().getEpochSecond();
            System.out.println("===================================== Generation " + (gen + 1) + " =====================================");
//            if (DEBUG) {
//                System.out.println(Arrays.toString(population.stream().map(Individual::getChromosome).toArray()));
//            }


            population = GA.evolve(population, trainSet, validationSet);

            // Найдём лучшую архитектуру
            Individual best = population.stream()
                .min(Comparator.comparing(Individual::getFitness))
                .orElseThrow();

            System.out.println("\nBest fitness: " + best.getFitness() + " for " + best.getChromosome());
            trainAndSaveResults(imagesTest, best, trainSet, validationSet, start, gen);

            //     new Thread(new FormDigits(neuralNetwork)).start();
        }
    }

    private static void trainAndSaveResults(List<Image> imagesTest, Individual best, List<Image> trainSet, List<Image> validationSet, long start, int gen) {
        NeuralNetwork neuralNetwork = buildNetworkFromChromosome(best.getChromosome());
        EPOCH_TRAINER.train(neuralNetwork, trainSet, validationSet);
        float trainAccuracy = neuralNetwork.test(trainSet);
        float validationAccuracy = neuralNetwork.test(validationSet);
        float testAccuracy = neuralNetwork.test(imagesTest);


//            if (DEBUG) {
//                System.out.println(neuralNetwork);
//            }

        long trainingTime = now().getEpochSecond() - start;
        printTimeTaken(trainingTime);

        int totalParams = neuralNetwork.getLayers().stream()
            .mapToInt(Layer::getParameterCount)
            .sum();

        ExcelLogger.saveResults(
            gen + 1,
            best.getFitness(),
            testAccuracy,
            validationAccuracy,
            trainAccuracy,
            totalParams,
            trainingTime,
            best.getChromosome().toString()
        );
    }


    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n", minutes, seconds);
    }
}

