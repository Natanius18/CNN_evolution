package natanius.thesis.cnn.evolution.genes;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PopulationGenerator {

    public static List<Individual> generateInitialPopulation(int populationSize, int convLayers, int maxFilters, int maxFilterSize, Random rand) {
        List<Individual> population = new ArrayList<>();

        for (int i = 0; i < populationSize; i++) {
            int[] chromosome = new int[convLayers * 2];

            // Генерация количества фильтров
            for (int j = 0; j < convLayers; j++) {
                chromosome[j] = rand.nextInt(maxFilters) + 1; // [1, maxFilters]
            }

            // Генерация размеров фильтров
            for (int j = 0; j < convLayers; j++) {
                chromosome[convLayers + j] = rand.nextInt(maxFilterSize) + 1; // [1, maxFilterSize]
            }

            population.add(new Individual(chromosome));
        }

        return population;
    }
}

