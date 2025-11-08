package natanius.thesis.cnn.evolution.genes;

import static java.lang.Math.floorDiv;
import static java.lang.Thread.currentThread;
import static java.time.Instant.now;
import static java.util.Comparator.comparingDouble;
import static natanius.thesis.cnn.evolution.data.Constants.CROSSOVER_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.ELITE_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.MUTANT_COUNT;
import static natanius.thesis.cnn.evolution.data.Constants.POPULATION_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.buildNetworkFromChromosome;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import lombok.RequiredArgsConstructor;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.network.EpochTrainer;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@RequiredArgsConstructor
public class GeneticAlgorithm {

    private final EpochTrainer epochTrainer = new EpochTrainer();

    public List<Individual> evolve(List<Individual> currentPopulation, List<Image> trainSet, List<Image> validationSet) {

        // 1. Оценка фитнеса
        AtomicInteger processedCount = new AtomicInteger(0);
        IntStream.range(0, currentPopulation.size())
            .parallel()
            .forEach(i -> {
                Individual ind = currentPopulation.get(i);
                if (ind.getFitness() == Float.MAX_VALUE) {
                    ind.setFitness(evaluateFitness(ind, trainSet, validationSet));
                }
                int processed = processedCount.incrementAndGet();
                String threadName = currentThread().getName();
                System.out.println("[" + processed + "/" + currentPopulation.size() + "], thread " + threadName.charAt(threadName.length() - 1) + ": " + ind);
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

    private float evaluateFitness(Individual ind, List<Image> trainSet, List<Image> validationSet) {
        try {
            long start = now().getEpochSecond();
            NeuralNetwork network = buildNetworkFromChromosome(ind.getChromosome());
            float accuracy = epochTrainer.train(network, trainSet, validationSet);
            long trainingTime = now().getEpochSecond() - start;
            printTimeTaken(trainingTime);
            
            int totalParams = network.getLayers().stream()
                .mapToInt(Layer::getParameterCount)
                .sum();
            
            float error = 100f - accuracy * 100f;
            float complexityPenalty = totalParams / 100_000f;
            return error + complexityPenalty;

        } catch (IllegalStateException e) {
            System.out.println("Invalid chromosome " + ind.getChromosome() + " → regenerating");
            ind.setChromosome(new Chromosome());
            return evaluateFitness(ind, trainSet, validationSet);
        }
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time for one: %d:%d ", minutes, seconds);
    }
}
