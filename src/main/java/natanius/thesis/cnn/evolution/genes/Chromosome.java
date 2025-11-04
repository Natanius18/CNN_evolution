package natanius.thesis.cnn.evolution.genes;

import static natanius.thesis.cnn.evolution.data.Constants.ACTIVATION_STRATEGIES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_CONV_STRIDES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FC_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTERS;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_FILTER_SIZES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_POOL_STRIDES;
import static natanius.thesis.cnn.evolution.data.Constants.ALLOWED_POOL_WINDOWS;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.MAX_FC_LAYERS;
import static natanius.thesis.cnn.evolution.data.Constants.MIN_CONV_BLOCKS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;
import static natanius.thesis.cnn.evolution.genes.LayerType.CONVOLUTION;
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

    /**
     * Генерує випадкову валідну CNN архітектуру для MNIST з дотриманням наступних правил:
     *
     * <p><b>СТРУКТУРА МЕРЕЖІ:</b>
     * <pre>
     * Input(28×28) → [Conv Block]+ → [FC Hidden]* → FC Output(10)
     * </pre>
     *
     * <p><b>ПРАВИЛА ДЛЯ CONVOLUTION BLOCKS:</b>
     * <ul>
     *   <li>Кількість блоків: {@value natanius.thesis.cnn.evolution.data.Constants#MIN_CONV_BLOCKS}
     *       - {@value natanius.thesis.cnn.evolution.data.Constants#MAX_CONV_BLOCKS}</li>
     *   <li><b>Monotonic filter growth:</b> Кількість фільтрів тільки збільшується або залишається константною
     *       (8 → 16 → 32, ніколи 32 → 16)</li>
     *   <li>Розміри фільтрів: 3×3, 5×5, 7×7</li>
     *   <li>Stride: 1 або 2 - випадково</li>
     *   <li>Padding: Same (filterSize/2) або Valid (0) - випадково</li>
     *   <li>Activation: ReLU, LeakyReLU, Sigmoid - випадково</li>
     * </ul>
     *
     * <p><b>ПРАВИЛА ДЛЯ MAX POOLING:</b>
     * <ul>
     *   <li>Розмір вікна: 2×2, stride: 2 (зменшує розмір вдвічі)</li>
     *   <li>Максимум pooling layers: 3 для MNIST 28×28</li>
     *   <li>Додається тільки якщо spatial розмір ≥ 6×6</li>
     *   <li>Ймовірність додавання: 50% після кожного Conv layer</li>
     * </ul>
     *
     * <p><b>ОБМЕЖЕННЯ НА SPATIAL РОЗМІРИ:</b>
     * <ul>
     *   <li>Мінімальний розмір після Conv: 3×3 (інакше блок пропускається)</li>
     *   <li>Формула Conv: output_size = (input_size - filter_size + 2×padding) / stride + 1</li>
     *   <li>Після pooling: output_size = input_size / 2</li>
     * </ul>
     *
     * <p><b>ПРАВИЛА ДЛЯ FULLY CONNECTED LAYERS:</b>
     * <ul>
     *   <li>Кількість FC layers: 1-{@value natanius.thesis.cnn.evolution.data.Constants#MAX_FC_LAYERS}
     *       (включаючи output)</li>
     *   <li>Hidden FC розміри: 64, 128, 256, 512 нейронів</li>
     *   <li>Hidden FC activation: ReLU, LeakyReLU, Sigmoid - випадково</li>
     *   <li><b>Output layer:</b> ЗАВЖДИ 10 нейронів + Linear activation (для MNIST)</li>
     * </ul>
     *
     * <p><b>ПРИКЛАДИ ЗГЕНЕРОВАНИХ АРХІТЕКТУР:</b>
     * <pre>
     * Conv(8,3×3,same,ReLU) → Pool → Conv(16,5×5,valid,Sigmoid) → FC(128,ReLU) → FC(10,Linear)
     * Conv(4,5×5,valid,ReLU) → Conv(8,3×3,same,ReLU) → Pool → Conv(16,3×3,same,ReLU) → FC(10,Linear)
     * Conv(16,3×3,same,LeakyReLU) → Pool → Conv(32,5×5,valid,ReLU) → Pool → FC(256,ReLU) → FC(10,Linear)
     * </pre>
     */
    public Chromosome() {
        layerGenes = new ArrayList<>();
        int numBlocks = MIN_CONV_BLOCKS + RANDOM.nextInt(MAX_CONV_BLOCKS - MIN_CONV_BLOCKS + 1);

        generateConvolutionBlocks(numBlocks);
        generateFullyConnectedLayers();
    }

    private void generateConvolutionBlocks(int numBlocks) {
        int currentFilters = ALLOWED_FILTERS[0];
        int spatialSize = 28;
        int poolingCount = 0;
        int maxPooling = 3;

        for (int i = 0; i < numBlocks; i++) {
            int numFilters = selectFiltersWithMonotonicGrowth(currentFilters);
            int filterSize = ALLOWED_FILTER_SIZES[RANDOM.nextInt(ALLOWED_FILTER_SIZES.length)];
            var activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
            int padding = RANDOM.nextBoolean() ? filterSize / 2 : 0;
            int convStride = ALLOWED_CONV_STRIDES[RANDOM.nextInt(ALLOWED_CONV_STRIDES.length)];

            int newSize = (spatialSize - filterSize + 2 * padding) / convStride + 1;
            if (newSize < 3) continue;
            spatialSize = newSize;
            currentFilters = numFilters;

            layerGenes.add(new LayerGene(CONVOLUTION, numFilters, filterSize, activation, padding, convStride));

            if (spatialSize >= 6 && poolingCount < maxPooling && RANDOM.nextBoolean()) {
                int poolWindow = ALLOWED_POOL_WINDOWS[RANDOM.nextInt(ALLOWED_POOL_WINDOWS.length)];
                int poolStride = ALLOWED_POOL_STRIDES[RANDOM.nextInt(ALLOWED_POOL_STRIDES.length)];
                layerGenes.add(new LayerGene(MAX_POOL, poolWindow, poolStride));
                spatialSize = (spatialSize - poolWindow) / poolStride + 1;
                poolingCount++;
            }
        }
    }

    private int selectFiltersWithMonotonicGrowth(int currentFilters) {
        int minFilterIndex = 0;
        for (int i = 0; i < ALLOWED_FILTERS.length; i++) {
            if (ALLOWED_FILTERS[i] >= currentFilters) {
                minFilterIndex = i;
                break;
            }
        }
        return ALLOWED_FILTERS[minFilterIndex + RANDOM.nextInt(ALLOWED_FILTERS.length - minFilterIndex)];
    }

    private void generateFullyConnectedLayers() {
        int numFCLayers = 1 + RANDOM.nextInt(MAX_FC_LAYERS);

        for (int i = 0; i < numFCLayers - 1; i++) {
            int fcSize = ALLOWED_FC_SIZES[RANDOM.nextInt(ALLOWED_FC_SIZES.length)];
            var activation = ACTIVATION_STRATEGIES[RANDOM.nextInt(ACTIVATION_STRATEGIES.length)];
            layerGenes.add(new LayerGene(FULLY_CONNECTED, fcSize, activation));
        }

        layerGenes.add(new LayerGene(FULLY_CONNECTED));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layerGenes.size(); i++) {
            sb.append(layerGenes.get(i));
            if (i < layerGenes.size() - 1) {
                sb.append(" → ");
            }
        }
        return sb.toString();
    }
}
