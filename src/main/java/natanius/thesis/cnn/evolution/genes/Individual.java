package natanius.thesis.cnn.evolution.genes;


import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Individual {

    private float fitness;
    private Chromosome chromosome;

    public Individual(Chromosome chromosome) {
        this.chromosome = chromosome;
        this.fitness = Float.MAX_VALUE; // по умолчанию — наихудшее значение
    }

    @Override
    public String toString() {
        return "Individual: {" + chromosome + ", Fitness: " + fitness + "}";
    }
}

