package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_CONV_STRIDES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_POOL_STRIDES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_POOL_WINDOWS;
import static natanius.thesis.cnn.evolution.data.Constants.LEARNING_RATE_FULLY_CONNECTED;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.data.Constants.getLearningRate;
import static natanius.thesis.cnn.evolution.genes.LayerType.CONVOLUTION;
import static natanius.thesis.cnn.evolution.genes.LayerType.FULLY_CONNECTED;
import static natanius.thesis.cnn.evolution.genes.LayerType.MAX_POOL;

import java.util.ArrayList;
import java.util.List;
import lombok.experimental.UtilityClass;
import natanius.thesis.cnn.evolution.activation.Linear;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

@UtilityClass
public class GeneticFunctions {

    // Кросовер: створюємо нащадка з двох батьків
    public static Chromosome crossover(Chromosome parent1, Chromosome parent2) {
        List<LayerGene> p1Layers = parent1.getLayerGenes();
        List<LayerGene> p2Layers = parent2.getLayerGenes();

        // Прибираємо FC шар для кросовера
        List<LayerGene> p1WithoutFC = p1Layers.subList(0, p1Layers.size() - 1);
        List<LayerGene> p2WithoutFC = p2Layers.subList(0, p2Layers.size() - 1);

        int minSize = Math.min(p1WithoutFC.size(), p2WithoutFC.size());
        int cutPoint = minSize > 1 ? RANDOM.nextInt(minSize) : 0;

        List<LayerGene> childLayers = new ArrayList<>();
        childLayers.addAll(p1WithoutFC.subList(0, cutPoint));
        childLayers.addAll(p2WithoutFC.subList(cutPoint, p2WithoutFC.size()));
        
        childLayers.add(new LayerGene(FULLY_CONNECTED));

        return new Chromosome(childLayers);
    }

    // Мутація: випадкова перестановка або заміна значень
    public static Chromosome mutate(Chromosome individual) {
        List<LayerGene> layers = new ArrayList<>(individual.getLayerGenes());
        // Прибираємо FC шар для мутації
        layers.removeLast();
        
        double mutationType = RANDOM.nextDouble();

        if (mutationType < 0.3) {
            // Змінити параметри випадкового Conv шару
            List<Integer> convIndices = new ArrayList<>();
            for (int i = 0; i < layers.size(); i++) {
                if (layers.get(i).getType() == CONVOLUTION) {
                    convIndices.add(i);
                }
            }
            if (!convIndices.isEmpty()) {
                int idx = convIndices.get(RANDOM.nextInt(convIndices.size()));
                int prevFilters = idx > 0 ? getPreviousConvFilters(layers, idx) : ALLOWED_FILTERS[0];
                int minFilterIndex = getMinFilterIndex(prevFilters);
                
                int filterSize = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
                int padding = RANDOM.nextBoolean() ? filterSize / 2 : 0;
                int convStride = ALLOWED_CONV_STRIDES[RANDOM.nextInt(ALLOWED_CONV_STRIDES.length)];
                layers.set(idx, new LayerGene(
                    CONVOLUTION,
                    ALLOWED_FILTERS[minFilterIndex + RANDOM.nextInt(ALLOWED_FILTERS.length - minFilterIndex)],
                    filterSize,
                    ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)],
                    padding,
                    convStride
                ));
            }
        } else if (mutationType < 0.5) {
            // Додати Conv шар
            int pos = RANDOM.nextInt(layers.size() + 1);
            int prevFilters = pos > 0 ? getPreviousConvFilters(layers, pos) : ALLOWED_FILTERS[0];
            int minFilterIndex = getMinFilterIndex(prevFilters);
            
            int filterSize = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
            int padding = RANDOM.nextBoolean() ? filterSize / 2 : 0;
            int convStride = ALLOWED_CONV_STRIDES[RANDOM.nextInt(ALLOWED_CONV_STRIDES.length)];
            layers.add(pos, new LayerGene(
                CONVOLUTION,
                ALLOWED_FILTERS[minFilterIndex + RANDOM.nextInt(ALLOWED_FILTERS.length - minFilterIndex)],
                filterSize,
                ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)],
                padding,
                convStride
            ));
        } else if (mutationType < 0.7) {
            // Додати/видалити MaxPool шар
            if (RANDOM.nextBoolean() && layers.size() > 1) {
                // Видалити випадковий MaxPool
                List<Integer> poolIndices = new ArrayList<>();
                for (int i = 0; i < layers.size(); i++) {
                    if (layers.get(i).getType() == MAX_POOL) {
                        poolIndices.add(i);
                    }
                }
                if (!poolIndices.isEmpty()) {
                    layers.remove((int) poolIndices.get(RANDOM.nextInt(poolIndices.size())));
                }
            } else {
                // Додати MaxPool після випадкового Conv
                List<Integer> convIndices = new ArrayList<>();
                for (int i = 0; i < layers.size(); i++) {
                    if (layers.get(i).getType() == CONVOLUTION) {
                        convIndices.add(i);
                    }
                }
                if (!convIndices.isEmpty()) {
                    int idx = convIndices.get(RANDOM.nextInt(convIndices.size()));
                    int poolWindow = ALLOWED_POOL_WINDOWS[RANDOM.nextInt(ALLOWED_POOL_WINDOWS.length)];
                    int poolStride = ALLOWED_POOL_STRIDES[RANDOM.nextInt(ALLOWED_POOL_STRIDES.length)];
                    layers.add(idx + 1, new LayerGene(MAX_POOL, poolWindow, poolStride));
                }
            }
        } else if (layers.size() > 2) {
            // Видалити випадковий шар (але не останній Conv)
            int idx = RANDOM.nextInt(layers.size() - 1);
            layers.remove(idx);
        }
        
        layers.add(new LayerGene(FULLY_CONNECTED));

        return new Chromosome(layers);
    }

    public static NeuralNetwork buildNetworkFromChromosome(Chromosome chromosome) {
        NetworkBuilder builder = new NetworkBuilder();

        for (LayerGene gene : chromosome.getLayerGenes()) {
            switch (gene.getType()) {
                case CONVOLUTION:
                    builder.addConvolutionLayer(
                        gene.getNumFilters(),
                        gene.getFilterSize(),
                        gene.getConvStride(),
                        getLearningRate(gene.getActivation()),
                        gene.getActivation(),
                        gene.getPadding()
                    );
                    break;
                    
                case MAX_POOL:
                    builder.addMaxPoolLayer(gene.getPoolWindow(), gene.getPoolStride());
                    break;
                    
                case FULLY_CONNECTED:
                    if (gene.getFcSize() != null) {
                        builder.addFullyConnectedLayer(
                            gene.getFcSize(),
                            LEARNING_RATE_FULLY_CONNECTED,
                            gene.getActivation()
                        );
                    } else {
                        builder.addFullyConnectedLayer(
                            LEARNING_RATE_FULLY_CONNECTED,
                            new Linear()
                        );
                    }
                    break;
            }
        }

        return builder.build();
    }

    private static int getPreviousConvFilters(List<LayerGene> layers, int currentIdx) {
        for (int i = currentIdx - 1; i >= 0; i--) {
            if (layers.get(i).getType() == CONVOLUTION) {
                return layers.get(i).getNumFilters();
            }
        }
        return ALLOWED_FILTERS[0];
    }

    private static int getMinFilterIndex(int minFilters) {
        for (int i = 0; i < ALLOWED_FILTERS.length; i++) {
            if (ALLOWED_FILTERS[i] >= minFilters) {
                return i;
            }
        }
        return ALLOWED_FILTERS.length - 1;
    }

}
