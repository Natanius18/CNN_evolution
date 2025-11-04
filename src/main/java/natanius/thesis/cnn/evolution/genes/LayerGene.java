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
    private final Integer fcSize;
    private final Integer poolWindow;
    private final Integer poolStride;
    private final Integer convStride;

    public LayerGene(LayerType type, Integer numFilters, Integer filterSize, Activation activation, int padding, int convStride) {
        this.type = type;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.activation = activation;
        this.padding = padding;
        this.fcSize = null;
        this.poolWindow = null;
        this.poolStride = null;
        this.convStride = convStride;
    }

    public LayerGene(LayerType type, Integer fcSize, Activation activation) {
        this.type = type;
        this.numFilters = null;
        this.filterSize = null;
        this.activation = activation;
        this.padding = 0;
        this.fcSize = fcSize;
        this.poolWindow = null;
        this.poolStride = null;
        this.convStride = null;
    }

    public LayerGene(LayerType type, int poolWindow, int poolStride) {
        this.type = type;
        this.numFilters = null;
        this.filterSize = null;
        this.activation = null;
        this.padding = 0;
        this.fcSize = null;
        this.poolWindow = poolWindow;
        this.poolStride = poolStride;
        this.convStride = null;
    }

    public LayerGene(LayerType type) {
        this.type = type;
        this.numFilters = null;
        this.filterSize = null;
        this.activation = null;
        this.padding = 0;
        this.fcSize = null;
        this.poolWindow = null;
        this.poolStride = null;
        this.convStride = null;
    }

    @Override
    public String toString() {
        switch (type) {
            case MAX_POOL -> {
                return String.format("MAX_POOL (%dx%d, stride=%d)", poolWindow, poolWindow, poolStride);
            }
            case FULLY_CONNECTED -> {
                String act = activation != null ? activation.getClass().getSimpleName() : "Linear";
                return fcSize != null ? String.format("FC(%d,%s)", fcSize, act) : "FC output";
            }
            case CONVOLUTION -> {
                String paddingType = padding == 0 ? "valid" : "same";
                return String.format("CONVOLUTION (%d filters %dx%d, stride=%d, %s padding + %s)",
                    numFilters, filterSize, filterSize, convStride, paddingType,
                    activation.getClass().getSimpleName());
            }
            default -> {
                return "UNKNOWN";
            }
        }
    }
}
