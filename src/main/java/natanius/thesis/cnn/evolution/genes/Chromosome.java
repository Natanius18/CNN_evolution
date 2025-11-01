package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.CONV_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.Arrays;
import lombok.AllArgsConstructor;
import lombok.Getter;
import natanius.thesis.cnn.evolution.activation.Activation;

@Getter
@AllArgsConstructor
public class Chromosome {

    private final int[] numFilters;
    private final int[] filterSizes;
    private final Activation activation;



    public Chromosome() {

        // Генерация количества фильтров
        numFilters = new int[CONV_LAYERS];
        for (int i = 0; i < CONV_LAYERS; i++) {
            numFilters[i] = ALLOWED_FILTERS[RANDOM.nextInt(ALLOWED_FILTERS.length)];
        }

        // Генерация размеров фильтров
        filterSizes = new int[CONV_LAYERS];
        for (int i = 0; i < CONV_LAYERS; i++) {
            filterSizes[i] = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
        }

        activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
    }

    @Override
    public String toString() {
        return "Chromosome{" +
            "numFilters=" + Arrays.toString(numFilters) +
            ", filterSizes=" + Arrays.toString(filterSizes) +
            ", activation=" + activation.getClass().getSimpleName() +
            '}';
    }
}
