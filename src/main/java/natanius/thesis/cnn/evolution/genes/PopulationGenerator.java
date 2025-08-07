package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.CONV_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_FILTER_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.POPULATION_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.ArrayList;
import java.util.List;
import lombok.experimental.UtilityClass;

@UtilityClass
public class PopulationGenerator {

    public static List<Individual> generateInitialPopulation() {
        List<Individual> population = new ArrayList<>();

        for (int i = 0; i < POPULATION_SIZE; i++) {
            int[] chromosome = new int[CONV_LAYERS * 2];

            // Генерация количества фильтров
            for (int j = 0; j < CONV_LAYERS; j++) {
                chromosome[j] = RANDOM.nextInt(MAX_FILTERS) + 1; // [1, maxFilters]
            }

            // Генерация размеров фильтров
            for (int j = 0; j < CONV_LAYERS; j++) {
                chromosome[CONV_LAYERS + j] = RANDOM.nextInt(MAX_FILTER_SIZE) + 1; // [1, maxFilterSize]
            }

            population.add(new Individual(chromosome));
        }

        return population;
    }
}

