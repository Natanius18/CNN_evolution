package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.CONV_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.CONV_STEP_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.DEBUG;
import static natanius.thesis.cnn.evolution.data.Constants.LEARNING_RATE;
import static natanius.thesis.cnn.evolution.data.Constants.LEARNING_RATE_FULLY_CONNECTED;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_POOL_STEP_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_POOL_WINDOW_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.Arrays;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class GeneticFunctions {

    // Кроссовер: создаём ребёнка из двух родителей
    public static Chromosome crossover(Chromosome parent1, Chromosome parent2) {
        int[] childFilters = new int[CONV_LAYERS];
        int[] childFilterSizes = new int[CONV_LAYERS];

        // Одноточечный кроссовер по количеству фильтров
        int cutPoint1 = RANDOM.nextInt(CONV_LAYERS);
        for (int i = 0; i < CONV_LAYERS; i++) {
            childFilters[i] = (i < cutPoint1) ? parent1.getNumFilters()[i] : parent2.getNumFilters()[i];
        }

        // Одноточечный кроссовер по размерам фильтров
        int cutPoint2 = RANDOM.nextInt(CONV_LAYERS);
        for (int i = 0; i < CONV_LAYERS; i++) {
            childFilterSizes[i] = (i < cutPoint2) ? parent1.getFilterSizes()[i] : parent2.getFilterSizes()[i];
        }

        // Активацию выбираем случайно у одного из родителей
        var childActivation = (RANDOM.nextBoolean()) ? parent1.getActivation() : parent2.getActivation();

        Chromosome child = new Chromosome(childFilters, childFilterSizes, childActivation);

        if (DEBUG) {
            System.out.println("\n--- Crossover ---");
            System.out.println("Parent1 filters: " + Arrays.toString(parent1.getNumFilters()));
            System.out.println("Parent2 filters: " + Arrays.toString(parent2.getNumFilters()));
            System.out.println("Child   filters: " + Arrays.toString(child.getNumFilters()));
            System.out.println("Child   sizes:   " + Arrays.toString(child.getFilterSizes()));
            System.out.println("Child   act:     " + child.getActivation().getClass().getSimpleName());
        }

        return child;
    }

    // Мутация: случайная перестановка или замена значений
    public static Chromosome mutate(Chromosome individual) {
        int[] filters = Arrays.copyOf(individual.getNumFilters(), CONV_LAYERS);
        int[] sizes = Arrays.copyOf(individual.getFilterSizes(), CONV_LAYERS);
        var activation = individual.getActivation();

        // Случайно меняем количество фильтров в одном месте
        int idxF = RANDOM.nextInt(CONV_LAYERS);
        filters[idxF] = ALLOWED_FILTERS[RANDOM.nextInt(ALLOWED_FILTERS.length)];

        // Случайно меняем размер фильтра в одном месте
        int idxS = RANDOM.nextInt(CONV_LAYERS);
        sizes[idxS] = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];

        // Иногда мутируем функцию активации
        if (RANDOM.nextDouble() < 0.3) { // вероятность 30%
            activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
        }

        Chromosome mutated = new Chromosome(filters, sizes, activation);

        if (DEBUG) {
            System.out.println("\n--- Mutation ---");
            System.out.println("Before filters: " + Arrays.toString(individual.getNumFilters()));
            System.out.println("After  filters: " + Arrays.toString(mutated.getNumFilters()));
            System.out.println("After  sizes:   " + Arrays.toString(mutated.getFilterSizes()));
            System.out.println("After  act:     " + mutated.getActivation().getClass().getSimpleName());
        }

        return mutated;
    }

    public static NeuralNetwork buildNetworkFromChromosome(Chromosome chromosome) {
        NetworkBuilder builder = new NetworkBuilder();

        // добавляем сверточные и pooling-слои
        for (int i = 0; i < CONV_LAYERS; i++) {
            int filters = chromosome.getNumFilters()[i];
            int filterSize = chromosome.getFilterSizes()[i];

            builder.addConvolutionLayer(
                filters,
                filterSize,
                CONV_STEP_SIZE,
                LEARNING_RATE
            ).addMaxPoolLayer(MAX_POOL_WINDOW_SIZE, MAX_POOL_STEP_SIZE);
        }

        // добавляем fully connected слой с активацией из хромосомы
        return builder
            .addFullyConnectedLayer(
                LEARNING_RATE_FULLY_CONNECTED,
                chromosome.getActivation()
            )
            .build();
    }

}
