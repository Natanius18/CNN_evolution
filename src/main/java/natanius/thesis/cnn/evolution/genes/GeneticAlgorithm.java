package natanius.thesis.cnn.evolution.genes;

import static java.util.Comparator.comparingDouble;
import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import lombok.RequiredArgsConstructor;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@RequiredArgsConstructor
public class GeneticAlgorithm {

    private final int convLayers;
    private final float mutationRate;
    private final int inputRows;
    private final int inputCols;
    private final int outputClasses;
    private final float learningRate;
    private final long seed;
    private final Random rand;

    private static final boolean FAST_MODE = true;

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
            Individual p1 = parents.get(rand.nextInt(parents.size()));
            Individual p2 = parents.get(rand.nextInt(parents.size()));

            int[] childChromosome = GeneticFunctions.crossover(p1.getChromosome(), p2.getChromosome(), convLayers, rand);

            if (rand.nextFloat() < mutationRate) {
                childChromosome = GeneticFunctions.mutate(childChromosome, convLayers, rand);
            }

            nextGeneration.add(new Individual(childChromosome));
        }

        return nextGeneration;
    }

    private float evaluateFitness(int[] chromosome, List<Image> imagesTrain, List<Image> imagesTest) {
        // Сборка сети по хромосоме
        NetworkBuilder builder = new NetworkBuilder(inputRows, inputCols, 25600);
        for (int i = 0; i < convLayers; i++) {
            int numFilters = chromosome[i];
            int filterSize = chromosome[i + convLayers];

            builder
                .addConvolutionLayer(numFilters, filterSize, 1, learningRate, seed)
                .addMaxPoolLayer(3, 2);
        }

        builder.addFullyConnectedLayer(outputClasses, 0.7, seed);
        NeuralNetwork network = builder.build();

        // === Изменение: теперь мы обучаем сеть перед тестом ===
        network.train(imagesTrain, 3); // ← можно настроить батч/эпохи как хочешь

        // После обучения — тестируем и возвращаем accuracy
        return network.test(imagesTest); // accuracy ∈ [0.0, 1.0]
    }
}
