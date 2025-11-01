package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.POPULATION_SIZE;

import java.util.ArrayList;
import java.util.List;
import lombok.experimental.UtilityClass;

@UtilityClass
public class PopulationGenerator {

    public static List<Individual> generateInitialPopulation() {
        List<Individual> population = new ArrayList<>();

        for (int i = 0; i < POPULATION_SIZE; i++) {
            population.add(new Individual(new Chromosome()));
        }

        return population;
    }
}

