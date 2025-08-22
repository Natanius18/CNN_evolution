package natanius.thesis.cnn.evolution.genes;

import static java.lang.Math.floorDiv;
import static java.util.Comparator.comparingDouble;
import static natanius.thesis.cnn.evolution.data.Constants.CROSSOVER_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.ELITE_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.MUTANT_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.POPULATION_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import lombok.RequiredArgsConstructor;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.network.EpochTrainer;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@RequiredArgsConstructor
public class GeneticAlgorithm {

    private final EpochTrainer epochTrainer = new EpochTrainer();

    public List<Individual> evolve(List<Individual> currentPopulation, List<Image> imagesTrain, List<Image> imagesTest) {

        // 1. Оценка фитнеса
        IntStream.range(0, currentPopulation.size())
            .parallel()
            .forEach(i -> {
                Individual ind = currentPopulation.get(i);
                if (ind.getFitness() == Float.MAX_VALUE) {
                    ind.setFitness(evaluateFitness(ind, imagesTrain, imagesTest));
                }
                System.out.println(Thread.currentThread().getName() + ": " + ind);
            });

        // 2. Сортировка по фитнесу
        currentPopulation.sort(comparingDouble(Individual::getFitness));

        int size = currentPopulation.size();

        // Элита
        List<Individual> nextGeneration = new ArrayList<>(currentPopulation.subList(0, ELITE_COUNT));

        // Потомки элиты
        while (nextGeneration.size() < ELITE_COUNT + CROSSOVER_COUNT) {
            Individual p1 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT));
            Individual p2 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT));
            int attempts = 0;
            while (p1.equals(p2)) {
                p2 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT));
                System.out.println("Trying another parent");
                attempts++;
                if (attempts > 10) {
                    p2 = currentPopulation.get(RANDOM.nextInt(ELITE_COUNT, POPULATION_SIZE));
                    break;
                }
            }

            Chromosome childChromosome = GeneticFunctions.crossover(
                p1.getChromosome(), p2.getChromosome());
            nextGeneration.add(new Individual(childChromosome));
        }

        // Мутанты
        while (nextGeneration.size() < ELITE_COUNT + CROSSOVER_COUNT + MUTANT_COUNT) {
            Individual base = currentPopulation.get(RANDOM.nextInt(size)); // может быть не из элиты
            Chromosome mutated = GeneticFunctions.mutate(base.getChromosome());
            nextGeneration.add(new Individual(mutated));
        }

        // Случайные иммигранты
        while (nextGeneration.size() < size) {
            nextGeneration.add(new Individual(new Chromosome()));
        }

        return nextGeneration;
    }

    private float evaluateFitness(Individual ind, List<Image> imagesTrain, List<Image> imagesTest) {
        try {
//            long start = now().getEpochSecond();
            NeuralNetwork network = buildNetworkFromChromosome(ind.getChromosome());
            float accuracy = epochTrainer.train(network, imagesTrain, imagesTest);
//            long trainingTime = now().getEpochSecond() - start;
//            printTimeTaken(trainingTime);
            return 100f - accuracy * 100f;  // чем меньше — тем лучше

        } catch (IllegalStateException e) {
//            if (DEBUG) {
            System.out.println("Invalid chromosome " + ind.getChromosome() + " → regenerating");
//            }
            ind.setChromosome(new Chromosome());
            return evaluateFitness(ind, imagesTrain, imagesTest);
        }
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time for one: %d:%d%n", minutes, seconds);
    }
}
