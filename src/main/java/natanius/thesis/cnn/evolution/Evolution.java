package natanius.thesis.cnn.evolution;

import static natanius.thesis.cnn.evolution.data.Constants.GENERATIONS;
import static natanius.thesis.cnn.evolution.data.Constants.TRAIN_AND_SAVE_BEST_MODEL;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;
import static natanius.thesis.cnn.evolution.genes.PopulationGenerator.generateInitialPopulation;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import natanius.thesis.cnn.evolution.genes.GeneticAlgorithm;
import natanius.thesis.cnn.evolution.genes.Individual;
import natanius.thesis.cnn.evolution.network.ExperimentalSandbox;

public class Evolution {

    public static void main(String[] args) {
        GeneticAlgorithm ga = new GeneticAlgorithm();

        List<Individual> population = generateInitialPopulation();
        ExperimentalSandbox sandbox = new ExperimentalSandbox();

        for (int gen = 0; gen < GENERATIONS; gen++) {
            System.out.println("===================================== Generation " + gen + " =====================================");
            System.out.println(Arrays.toString(population.stream().map(individual -> Arrays.toString(individual.getChromosome())).toArray()));
            population = ga.evolve(population, sandbox.getImagesTrain(), sandbox.getImagesTest());

            // Найдём лучшую архитектуру
            Individual best = population.stream()
                .min(Comparator.comparing(Individual::getFitness))
                .orElseThrow();

            System.out.println("Best fitness: " + best.getFitness());
            System.out.println("Chromosome: " + Arrays.toString(best.getChromosome()));
            System.out.println(buildNetworkFromChromosome(best.getChromosome()));

            if (TRAIN_AND_SAVE_BEST_MODEL) {
                sandbox.checkArchitecture(gen, buildNetworkFromChromosome(best.getChromosome()), best.getChromosome());
            }
        }
    }
}

