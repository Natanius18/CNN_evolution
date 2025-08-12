package natanius.thesis.cnn.evolution.genes;

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

    // Создание потомка, взяв части генов у двух родителей
    public static int[] crossover(int[] parent1, int[] parent2) {
        if (DEBUG) {
            System.out.println("\n--- Crossover ---");
            System.out.println("Parent 1: " + Arrays.toString(parent1));
            System.out.println("Parent 2: " + Arrays.toString(parent2));
        }

        int[] child = Arrays.copyOf(parent2, parent2.length);

        // Кроссовер по числу фильтров
        int p1 = RANDOM.nextInt(CONV_LAYERS);
        int p2 = RANDOM.nextInt(CONV_LAYERS);
        int start1 = Math.min(p1, p2);
        int end1 = Math.max(p1, p2);
        if (end1 + CONV_STEP_SIZE - start1 >= 0) {
            System.arraycopy(parent1, start1, child, start1, end1 + CONV_STEP_SIZE - start1);
        }

        // Кроссовер по размерам фильтров
        int p3 = RANDOM.nextInt(CONV_LAYERS) + CONV_LAYERS;
        int p4 = RANDOM.nextInt(CONV_LAYERS) + CONV_LAYERS;
        int start2 = Math.min(p3, p4);
        int end2 = Math.max(p3, p4);
        if (end2 + CONV_STEP_SIZE - start2 >= 0) {
            System.arraycopy(parent1, start2, child, start2, end2 + CONV_STEP_SIZE - start2);
        }

        if (DEBUG) {
            System.out.println("Child:    " + Arrays.toString(child));
        }
        return child;
    }

    // Перестановка двух случайных фильтров и размеров
    public static int[] mutate(int[] individual) {
        if (DEBUG) {
            System.out.println("\n--- Mutation ---");
            System.out.println("Before: " + Arrays.toString(individual));
        }

        int[] child = Arrays.copyOf(individual, individual.length);

        // Мутация фильтров (перестановка)
        int i1 = RANDOM.nextInt(CONV_LAYERS);
        int i2 = RANDOM.nextInt(CONV_LAYERS);
        int temp = child[i1];
        child[i1] = child[i2];
        child[i2] = temp;

        // Мутация размеров фильтров (перестановка)
        int j1 = RANDOM.nextInt(CONV_LAYERS) + CONV_LAYERS;
        int j2 = RANDOM.nextInt(CONV_LAYERS) + CONV_LAYERS;
        temp = child[j1];
        child[j1] = child[j2];
        child[j2] = temp;

        if (DEBUG) {
            System.out.println("After:  " + Arrays.toString(child));
        }
        return child;
    }

    public static NeuralNetwork buildNetworkFromChromosome(int[] chromosome) {
        NetworkBuilder builder = new NetworkBuilder();

        for (int i = 0; i < CONV_LAYERS; i++) {
            int filters = chromosome[i];
            int filterSize = chromosome[i + CONV_LAYERS];

            builder.addConvolutionLayer(filters, filterSize, CONV_STEP_SIZE, LEARNING_RATE)
                .addMaxPoolLayer(MAX_POOL_WINDOW_SIZE, MAX_POOL_STEP_SIZE);
        }

        return builder
            .addFullyConnectedLayer(LEARNING_RATE_FULLY_CONNECTED)
            .build();
    }
}
