package natanius.thesis.cnn.evolution.genes;


import java.util.Arrays;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Individual {

    private float fitness;
    private int[] chromosome;

    public Individual(int[] chromosome) {
        this.chromosome = chromosome;
        this.fitness = Float.MAX_VALUE; // по умолчанию — наихудшее значение
    }

    public Individual copy() {
        return new Individual(chromosome.clone());
    }

    @Override
    public String toString() {
        return "Individual[Chromosome: " + Arrays.toString(chromosome) + ", Fitness: " + fitness + "]";
    }
}

