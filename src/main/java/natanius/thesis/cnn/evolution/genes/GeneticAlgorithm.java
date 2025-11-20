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
import static natanius.thesis.cnn.evolution.genes.GeneticFunctions.crossover;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    public static final Map<String, Float> CACHE = new HashMap<>();

    public List<Individual> evolve(List<Individual> currentPopulation, List<Image> trainSet, List<Image> validationSet) {
        checkCache(currentPopulation);

        evaluateFitnessForAll(currentPopulation, trainSet, validationSet);

        currentPopulation.sort(comparingDouble(Individual::getFitness));

        List<Individual> nextGeneration = new ArrayList<>(currentPopulation.subList(0, ELITE_COUNT));
        addChildrenOfElite(currentPopulation, nextGeneration);
        addMutants(currentPopulation, nextGeneration);
        addRandomImmigrants(nextGeneration);
        System.out.println("Cache size: " + CACHE.size());
        return nextGeneration;
    }

    private static void checkCache(List<Individual> currentPopulation) {
        for (Individual ind : currentPopulation) {
            while (ind.getFitness() == Float.MAX_VALUE && CACHE.containsKey(ind.getChromosome().toString())) {
                System.out.println("Already checked chromosome {" + ind.getChromosome().toString() + "}, generating a new one");
                ind.setChromosome(new Chromosome());
            }
            CACHE.putIfAbsent(ind.getChromosome().toString(), null);
        }
    }

    private void evaluateFitnessForAll(List<Individual> currentPopulation, List<Image> trainSet, List<Image> validationSet) {
        AtomicInteger processedCount = new AtomicInteger(0);
        IntStream.range(0, currentPopulation.size())
            .parallel()
            .forEach(i -> {
                Individual ind = currentPopulation.get(i);
                if (ind.getFitness() == Float.MAX_VALUE) {
                    float fitness = evaluateFitness(ind, trainSet, validationSet);
                    ind.setFitness(fitness);
                    CACHE.put(ind.getChromosome().toString(), fitness);
                }
                int processed = processedCount.incrementAndGet();
                String threadName = currentThread().getName();
                String[] split = threadName.split("-");
                System.out.println("[" + processed + "/" + currentPopulation.size() + "], thread " +
                    (split.length > 1 ? split[split.length - 1] : "0") + ": " + ind);
            });
    }

    private static void addChildrenOfElite(List<Individual> currentPopulation, List<Individual> nextGeneration) {
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
            Chromosome childChromosome = crossover(p1.getChromosome(), p2.getChromosome());
            nextGeneration.add(new Individual(childChromosome));
        }
    }

    private static void addMutants(List<Individual> currentPopulation, List<Individual> nextGeneration) {
        while (nextGeneration.size() < ELITE_COUNT + CROSSOVER_COUNT + MUTANT_COUNT) {
            Individual base = currentPopulation.get(RANDOM.nextInt(POPULATION_SIZE)); // може бути не з еліти
            Chromosome mutated = GeneticFunctions.mutate(base.getChromosome());
            nextGeneration.add(new Individual(mutated));
        }
    }

    private static void addRandomImmigrants(List<Individual> nextGeneration) {
        while (nextGeneration.size() < POPULATION_SIZE) {
            nextGeneration.add(new Individual(new Chromosome()));
        }
    }

    private float evaluateFitness(Individual ind, List<Image> trainSet, List<Image> validationSet) {
        try {
            long start = now().getEpochSecond();
            NeuralNetwork network = buildNetworkFromChromosome(ind.getChromosome());
            float accuracy = epochTrainer.train(network, trainSet, validationSet);
            long trainingTime = now().getEpochSecond() - start;
            printTimeTaken(trainingTime);
            return countFitness(network, accuracy);

        } catch (IllegalStateException e) {
            System.out.println("Invalid chromosome " + ind.getChromosome() + " → regenerating");
            Chromosome chromosome = new Chromosome();
            while (CACHE.containsKey(chromosome.toString())) {
                chromosome = new Chromosome();
            }
            ind.setChromosome(chromosome);
            CACHE.put(ind.getChromosome().toString(), null);
            return evaluateFitness(ind, trainSet, validationSet);
        }
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time for one: %d:%d ", minutes, seconds);
    }

    private static float countFitness(NeuralNetwork network, float accuracy) {
        int totalParams = network.getLayers().stream()
            .mapToInt(Layer::getParameterCount)
            .sum();

        float error = 100f - accuracy * 100f;
        float complexityPenalty = totalParams / 100_000f;
        return error + complexityPenalty;
    }
}
