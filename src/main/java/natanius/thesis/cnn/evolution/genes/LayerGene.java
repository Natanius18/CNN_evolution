package natanius.thesis.cnn.evolution.genes;

import lombok.Getter;
import natanius.thesis.cnn.evolution.activation.Activation;

@Getter
public class LayerGene {
    private final LayerType type;
    private final Integer numFilters;
    private final Integer filterSize;
    private final Activation activation;
    private final int padding;

    public LayerGene(LayerType type, Integer numFilters, Integer filterSize, Activation activation, int padding) {
        this.type = type;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.activation = activation;
        this.padding = padding;
    }

    public LayerGene(LayerType type) {
        this.type = type;
        this.numFilters = null;
        this.filterSize = null;
        this.activation = null;
        this.padding = 0;
    }

    @Override
    public String toString() {
        if (type == LayerType.MAX_POOL) {
            return "MAX_POOL";
        } else if (type == LayerType.FULLY_CONNECTED) {
            return "FC";
        }
        String paddingType = padding == 0 ? "valid" : "same";
        return String.format("%s(%d filters %dx%d, %s padding + %s)",
            type, numFilters, filterSize, filterSize, paddingType,
            activation.getClass().getSimpleName());
    }
}
