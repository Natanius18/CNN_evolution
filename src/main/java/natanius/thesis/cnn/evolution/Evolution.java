package natanius.thesis.cnn.evolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static natanius.thesis.cnn.evolution.data.Constants.DATASET_FRACTION;
import static natanius.thesis.cnn.evolution.data.Constants.DEBUG;
import static natanius.thesis.cnn.evolution.data.Constants.FAST_MODE;
import static natanius.thesis.cnn.evolution.data.Constants.GENERATIONS;
import static natanius.thesis.cnn.evolution.data.Constants.TRAIN_AND_SAVE_BEST_MODEL;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;
import static natanius.thesis.cnn.evolution.genes.PopulationGenerator.generateInitialPopulation;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.genes.GeneticAlgorithm;
import natanius.thesis.cnn.evolution.genes.Individual;
import natanius.thesis.cnn.evolution.network.ExperimentalSandbox;

public class Evolution {

    public static void main(String[] args) {
        GeneticAlgorithm ga = new GeneticAlgorithm();

        List<Individual> population = generateInitialPopulation();
        ExperimentalSandbox sandbox = new ExperimentalSandbox();
        List<Image> imagesTrain = sandbox.getImagesTrain();
        List<Image> imagesTest = sandbox.getImagesTest();
        if (FAST_MODE) {
            imagesTrain = imagesTrain.subList(0, (int) (imagesTrain.size() * DATASET_FRACTION));
            imagesTest =  imagesTest.subList(0, (int) (imagesTest.size() * DATASET_FRACTION));
            System.out.println("Sizes: " + imagesTrain.size() + " " + imagesTest.size());
        }

        for (int gen = 0; gen < GENERATIONS; gen++) {
            long start = now().getEpochSecond();
            System.out.println("===================================== Generation " + (gen + 1) + " =====================================");
            if (DEBUG) {
                System.out.println(Arrays.toString(population.stream().map(individual -> Arrays.toString(individual.getChromosome())).toArray()));
            }



            population = ga.evolve(population, imagesTrain, imagesTest);

            // Найдём лучшую архитектуру
            Individual best = population.stream()
                .min(Comparator.comparing(Individual::getFitness))
                .orElseThrow();

            System.out.println("\nBest fitness: " + best.getFitness() + " for " + Arrays.toString(best.getChromosome()));
            if (DEBUG) {
                System.out.println(buildNetworkFromChromosome(best.getChromosome()));
            }
            long trainingTime = now().getEpochSecond() - start;
            printTimeTaken(trainingTime);

            if (TRAIN_AND_SAVE_BEST_MODEL) {
//                sandbox.train(gen, buildNetworkFromChromosome(best.getChromosome()), best.getChromosome());
            }
        }
    }


    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n", minutes, seconds);
    }
}

