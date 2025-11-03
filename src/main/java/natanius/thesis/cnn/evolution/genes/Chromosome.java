package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.MIN_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

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

        for (int i = 0; i < numBlocks; i++) {
            int numFilters = ALLOWED_FILTERS[RANDOM.nextInt(ALLOWED_FILTERS.length)];
            int filterSize = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
            var activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
            int padding = RANDOM.nextBoolean() ? filterSize / 2 : 0;  // same or valid

            layerGenes.add(new LayerGene(LayerType.CONVOLUTION, numFilters, filterSize, activation, padding));
            
            if (RANDOM.nextBoolean()) {
                layerGenes.add(new LayerGene(LayerType.MAX_POOL));
            }
        }
        
        layerGenes.add(new LayerGene(LayerType.FULLY_CONNECTED));
    }

    @Override
    public String toString() {
        return "Chromosome{layerGenes=" + layerGenes + '}';
    }
}
