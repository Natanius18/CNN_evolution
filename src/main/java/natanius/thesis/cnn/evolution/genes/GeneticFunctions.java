package natanius.thesis.cnn.evolution.genes;

import java.util.Arrays;
import java.util.Random;

public class GeneticFunctions {

    // Создание потомка, взяв части генов у двух родителей
    public static int[] crossover(int[] parent1, int[] parent2, int convLayers, Random rand) {
        System.out.println("\n--- Crossover ---");
        System.out.println("Parent 1: " + Arrays.toString(parent1));
        System.out.println("Parent 2: " + Arrays.toString(parent2));

        int[] child = Arrays.copyOf(parent2, parent2.length);

        // Кроссовер по числу фильтров
        int p1 = rand.nextInt(convLayers);
        int p2 = rand.nextInt(convLayers);
        int start1 = Math.min(p1, p2);
        int end1 = Math.max(p1, p2);
        if (end1 + 1 - start1 >= 0) {
            System.arraycopy(parent1, start1, child, start1, end1 + 1 - start1);
        }

        // Кроссовер по размерам фильтров
        int p3 = rand.nextInt(convLayers) + convLayers;
        int p4 = rand.nextInt(convLayers) + convLayers;
        int start2 = Math.min(p3, p4);
        int end2 = Math.max(p3, p4);
        if (end2 + 1 - start2 >= 0) {
            System.arraycopy(parent1, start2, child, start2, end2 + 1 - start2);
        }

        System.out.println("Child:    " + Arrays.toString(child));
        return child;
    }

    // Перестановка двух случайных фильтров и размеров
    public static int[] mutate(int[] individual, int convLayers, Random rand) {
        System.out.println("\n--- Mutation ---");
        System.out.println("Before: " + Arrays.toString(individual));

        int[] child = Arrays.copyOf(individual, individual.length);

        // Мутация фильтров (перестановка)
        int i1 = rand.nextInt(convLayers);
        int i2 = rand.nextInt(convLayers);
        int temp = child[i1];
        child[i1] = child[i2];
        child[i2] = temp;

        // Мутация размеров фильтров (перестановка)
        int j1 = rand.nextInt(convLayers) + convLayers;
        int j2 = rand.nextInt(convLayers) + convLayers;
        temp = child[j1];
        child[j1] = child[j2];
        child[j2] = temp;

        System.out.println("After:  " + Arrays.toString(child));
        return child;
    }
}
