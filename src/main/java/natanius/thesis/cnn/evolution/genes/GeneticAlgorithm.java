package natanius.thesis.cnn.evolution.genes;

import static java.util.Comparator.comparingDouble;
import static natanius.thesis.cnn.evolution.data.Constants.BATCH_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.CROSSOVER_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.ELITE_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.FAST_MODE;
import static natanius.thesis.cnn.evolution.data.Constants.MUTANT_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;
import static natanius.thesis.cnn.evolution.genes.PopulationGenerator.generateChromosome;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.RequiredArgsConstructor;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@RequiredArgsConstructor
public class GeneticAlgorithm {
    public List<Individual> evolve(List<Individual> currentPopulation, List<Image> imagesTrain, List<Image> imagesTest) {

        if (FAST_MODE) {
            imagesTrain = imagesTrain.subList(0, 200);
        }

        // 1. Оценка фитнеса
        for (int i = 0, currentPopulationSize = currentPopulation.size(); i < currentPopulationSize; i++) {
            Individual ind = currentPopulation.get(i);
            System.out.println("Individual " + i);
            if (ind.getFitness() == Float.MAX_VALUE) {
                float accuracy = evaluateFitness(ind, imagesTrain, imagesTest);
                ind.setFitness(100f - accuracy * 100f); // чем меньше — тем лучше
            }
        }

        // 2. Сортировка по фитнесу
        currentPopulation.sort(comparingDouble(Individual::getFitness));

        int size = currentPopulation.size();

        // Элита
        List<Individual> nextGeneration = new ArrayList<>(currentPopulation.subList(0, ELITE_COUNT));

        // Потомки элиты
        while (nextGeneration.size() < ELITE_COUNT + CROSSOVER_COUNT) {
            Individual p1 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT));
            Individual p2 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT));

            int[] childChromosome = GeneticFunctions.crossover(
                p1.getChromosome(), p2.getChromosome());
            nextGeneration.add(new Individual(childChromosome));
        }

        // Мутанты
        while (nextGeneration.size() < ELITE_COUNT + CROSSOVER_COUNT + MUTANT_COUNT) {
            Individual base = currentPopulation.get(RANDOM.nextInt(size)); // может быть не из элиты
            int[] mutated = GeneticFunctions.mutate(base.getChromosome());
            nextGeneration.add(new Individual(mutated));
        }

        // Случайные иммигранты
        while (nextGeneration.size() < size) {
            nextGeneration.add(new Individual(generateChromosome()));
        }

        return nextGeneration;
    }

    private float evaluateFitness(Individual ind, List<Image> imagesTrain, List<Image> imagesTest) {
        try {
            NeuralNetwork network = buildNetworkFromChromosome(ind.getChromosome());

            network.train(imagesTrain, BATCH_SIZE);
            return network.test(imagesTest);

        } catch (IllegalStateException e) {
            System.out.println("Invalid chromosome " + Arrays.toString(ind.getChromosome()) + " → regenerating");

            int[] newChromosome = generateChromosome();
            ind.setChromosome(newChromosome);
            return evaluateFitness(ind, imagesTrain, imagesTest);
        }
    }

}
