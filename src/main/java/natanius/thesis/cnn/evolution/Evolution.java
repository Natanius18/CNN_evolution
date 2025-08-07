package natanius.thesis.cnn.evolution;

import static natanius.thesis.cnn.evolution.data.DataReader.loadTestData;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTrainData;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.genes.GeneticAlgorithm;
import natanius.thesis.cnn.evolution.genes.Individual;
import natanius.thesis.cnn.evolution.genes.PopulationGenerator;
import natanius.thesis.cnn.evolution.network.ExperimentalSandbox;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class Evolution {

    public static void main(String[] args) {
        // === Настройки ===
        final int convLayers = 1;
        final int populationSize = 10;
        final int generations = 5;
        final int maxFilters = 32;
        final int maxFilterSize = 7;
        final float mutationRate = 0.2f;
        final float learningRate = 0.3f;
        final long seed = 123;
        final int inputRows = 28;
        final int inputCols = 28;
        final int outputClasses = 10;

        List<Image> imagesTrain = loadTrainData();
        List<Image> imagesTest = loadTestData();

        Random rand = new Random(seed);
        GeneticAlgorithm ga = new GeneticAlgorithm(
            convLayers, mutationRate,
            inputRows, inputCols,
            outputClasses, learningRate,
            seed, rand
        );

        // === Инициализация популяции ===
        List<Individual> population = PopulationGenerator.generateInitialPopulation(
            populationSize, convLayers, maxFilters, maxFilterSize, rand
        );

        for (int gen = 0; gen < generations; gen++) {
            System.out.println("=== Generation " + gen + " ===");
            System.out.println(Arrays.toString(population.stream().map(individual -> Arrays.toString(individual.getChromosome())).toArray()));
            population = ga.evolve(population, imagesTrain, imagesTest);

            // Найдём лучшую архитектуру
            Individual best = population.stream()
                .min(Comparator.comparing(Individual::getFitness))
                .orElseThrow();

            System.out.println("Best fitness: " + best.getFitness());
            System.out.println("Chromosome: " + Arrays.toString(best.getChromosome()));

            // === Тренируем лучшую сеть ===
            NeuralNetwork bestNet = buildNetworkFromChromosome(best.getChromosome(), convLayers, inputRows, inputCols, outputClasses, learningRate, seed);
            ExperimentalSandbox sandbox = new ExperimentalSandbox();
            sandbox.checkArchitecture(gen, bestNet, best.getChromosome());
        }
    }

    private static NeuralNetwork buildNetworkFromChromosome(int[] chromosome, int convLayers,
                                                            int inputRows, int inputCols,
                                                            int outputClasses, float learningRate, long seed) {
        NetworkBuilder builder = new NetworkBuilder(inputRows, inputCols, 25600);

        for (int i = 0; i < convLayers; i++) {
            int filters = chromosome[i];
            int filterSize = chromosome[i + convLayers];

            builder.addConvolutionLayer(filters, filterSize, 1, learningRate, seed)
                .addMaxPoolLayer(2, 2);
        }

        return builder.addFullyConnectedLayer(outputClasses, learningRate, seed)
            .build();
    }
}

