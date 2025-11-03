package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FC_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_FC_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.MIN_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.LayerType.FULLY_CONNECTED;
import static natanius.thesis.cnn.evolution.genes.LayerType.MAX_POOL;

import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class Chromosome {

    private final List<LayerGene> layerGenes;

    public Chromosome() {
        int numBlocks = MIN_CONV_BLOCKS + RANDOM.nextInt(MAX_CONV_BLOCKS - MIN_CONV_BLOCKS + 1);
        layerGenes = new ArrayList<>();

        int currentFilters = ALLOWED_FILTERS[0];  // Start with minimum
        int spatialSize = 28;  // MNIST input size
        int poolingCount = 0;
        int maxPooling = 2;  // Max 2-3 pooling for 28x28

        for (int i = 0; i < numBlocks; i++) {
            int minFilterIndex = 0;
            for (int j = 0; j < ALLOWED_FILTERS.length; j++) {
                if (ALLOWED_FILTERS[j] >= currentFilters) {
                    minFilterIndex = j;
                    break;
                }
            }
            int numFilters = ALLOWED_FILTERS[minFilterIndex + RANDOM.nextInt(ALLOWED_FILTERS.length - minFilterIndex)];
            currentFilters = numFilters;

            int filterSize = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
            var activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
            int padding = RANDOM.nextBoolean() ? filterSize / 2 : 0;

            int newSize = (spatialSize - filterSize + 2 * padding) + 1;
            if (newSize < 3) continue;
            spatialSize = newSize;

            layerGenes.add(new LayerGene(LayerType.CONVOLUTION, numFilters, filterSize, activation, padding));
            
            if (spatialSize >= 6 && poolingCount < maxPooling && RANDOM.nextBoolean()) {
                layerGenes.add(new LayerGene(MAX_POOL));
                spatialSize = spatialSize / 2;
                poolingCount++;
            }
        }
        
        // Add FC layers (1-3 layers, last is always output with Linear)
        int numFCLayers = 1 + RANDOM.nextInt(MAX_FC_LAYERS);
        for (int i = 0; i < numFCLayers - 1; i++) {
            int fcSize = ALLOWED_FC_SIZES[RANDOM.nextInt(ALLOWED_FC_SIZES.length)];
            var activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
            layerGenes.add(new LayerGene(FULLY_CONNECTED, fcSize, activation));
        }
        layerGenes.add(new LayerGene(FULLY_CONNECTED));  // Output layer
    }

    @Override
    public String toString() {
        return "Chromosome{layerGenes=" + layerGenes + '}';
    }
}
