package natanius.thesis.cnn.evolution.genes;

import static java.util.Comparator.comparingDouble;
import static java.util.stream.Collectors.toList;
import static natanius.thesis.cnn.evolution.data.Constants.BATCH_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.CONV_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.FAST_MODE;
import static natanius.thesis.cnn.evolution.data.Constants.MUTATION_RATE;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;

import java.util.ArrayList;
import java.util.List;
import lombok.RequiredArgsConstructor;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@RequiredArgsConstructor
public class GeneticAlgorithm {
    public List<Individual> evolve(List<Individual> currentPopulation, List<Image> imagesTrain, List<Image> imagesTest) {

        if (FAST_MODE) {
            imagesTrain = imagesTrain.stream()
                .limit(200)
                .collect(toList());
        }

        // 1. Оценка фитнеса
        for (Individual ind : currentPopulation) {
            if (ind.getFitness() == Float.MAX_VALUE) {
                float accuracy = evaluateFitness(ind.getChromosome(), imagesTrain, imagesTest);
                ind.setFitness(100f - accuracy * 100f); // чем меньше — тем лучше
            }
        }

        // 2. Отбор лучших (top 50%)
        currentPopulation.sort(comparingDouble(Individual::getFitness));
        List<Individual> parents = currentPopulation.subList(0, currentPopulation.size() / 2);

        // 3. Скрещивание и мутация → новая популяция
        List<Individual> nextGeneration = new ArrayList<>(parents);
        while (nextGeneration.size() < currentPopulation.size()) {
            Individual p1 = parents.get(RANDOM.nextInt(parents.size()));
            Individual p2 = parents.get(RANDOM.nextInt(parents.size()));

            int[] childChromosome = GeneticFunctions.crossover(p1.getChromosome(), p2.getChromosome(), CONV_LAYERS);

            if (RANDOM.nextFloat() < MUTATION_RATE) {
                childChromosome = GeneticFunctions.mutate(childChromosome, CONV_LAYERS);
            }

            nextGeneration.add(new Individual(childChromosome));
        }

        return nextGeneration;
    }

    private float evaluateFitness(int[] chromosome, List<Image> imagesTrain, List<Image> imagesTest) {
        NeuralNetwork network = buildNetworkFromChromosome(chromosome);

        network.train(imagesTrain, BATCH_SIZE);

        return network.test(imagesTest); // accuracy ∈ [0.0, 1.0]
    }
}
