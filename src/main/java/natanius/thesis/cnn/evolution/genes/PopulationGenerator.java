package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.CONV_LAYERS;
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
            population.add(new Individual(generateChromosome()));
        }

        return population;
    }

    public static int[] generateChromosome() {
        int[] chromosome = new int[CONV_LAYERS * 2];

        // Генерация количества фильтров
        for (int j = 0; j < CONV_LAYERS; j++) {
            chromosome[j] = ALLOWED_FILTERS[RANDOM.nextInt(ALLOWED_FILTERS.length)];
        }

        // Генерация размеров фильтров
        for (int j = 0; j < CONV_LAYERS; j++) {
            chromosome[CONV_LAYERS + j] = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
        }

        return chromosome;
    }
}

