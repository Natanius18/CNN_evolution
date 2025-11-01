package natanius.thesis.cnn.evolution;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.activation.Linear;
import natanius.thesis.cnn.evolution.activation.ReLU;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.layers.ConvolutionLayer;
import natanius.thesis.cnn.evolution.layers.FullyConnectedLayer;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;

public class WeightUpdateTest {

    public static void testWeightUpdates(List<Image> images) {
        System.out.println("🔬 WEIGHT UPDATE DIAGNOSTIC TEST");
        System.out.println("═".repeat(60));

        List<Image> testSet = images.subList(0, 10);

        NeuralNetwork network = new NetworkBuilder()
            .addConvolutionLayer(8, 3, 1, 0.01, new ReLU(), 1)
            .addMaxPoolLayer(2, 2)
            .addFullyConnectedLayer(0.01, new Linear())
            .build();

        // Находим слои
        ConvolutionLayer convLayer = null;
        FullyConnectedLayer fcLayer = null;

        for (var layer : network.getLayers()) {
            if (layer instanceof ConvolutionLayer && convLayer == null) {
                convLayer = (ConvolutionLayer) layer;
            }
            if (layer instanceof FullyConnectedLayer) {
                fcLayer = (FullyConnectedLayer) layer;
            }
        }

        if (convLayer == null || fcLayer == null) {
            System.out.println("❌ Layers not found!");
            return;
        }

        // ═══════════════════════════════════════════════════════════
        // ЧАСТЬ 1: Сохраняем начальные веса
        // ═══════════════════════════════════════════════════════════
        System.out.println("\n📊 PART 1: Saving initial weights...");

        // Conv weights
        double[][] initialConvWeights = new double[8][9];
        for (int f = 0; f < 8; f++) {
            double[][] filter = convLayer.getFilters().get(f);
            int idx = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    initialConvWeights[f][idx++] = filter[i][j];
                }
            }
        }

        // FC weights - сохраняем ВСЕ веса
        int fcRows = fcLayer.getWeights().length;
        int fcCols = fcLayer.getWeights()[0].length;
        double[][] initialFCWeights = new double[fcRows][fcCols];

        for (int i = 0; i < fcRows; i++) {
            System.arraycopy(fcLayer.getWeights()[i], 0, initialFCWeights[i], 0, fcCols);
        }

        System.out.printf("   Saved Conv weights: 8 filters × 9 weights = 72 values%n");
        System.out.printf("   Saved FC weights: %d × %d = %d values%n",
            fcRows, fcCols, fcRows * fcCols);

        // Выводим несколько примеров
        System.out.println("\n   Sample initial Conv weights:");
        for (int f = 0; f < 3; f++) {
            System.out.printf("     Filter[%d][0-2]: [%.6f, %.6f, %.6f]%n",
                f, initialConvWeights[f][0], initialConvWeights[f][1], initialConvWeights[f][2]);
        }

        System.out.println("\n   Sample initial FC weights (first rows):");
        for (int i = 0; i < Math.min(3, fcRows); i++) {
            System.out.printf("     FC[%d][0-2]: [%.6f, %.6f, %.6f]%n",
                i, initialFCWeights[i][0], initialFCWeights[i][1], initialFCWeights[i][2]);
        }

        System.out.println("\n   Sample initial FC weights (last rows):");
        for (int i = Math.max(0, fcRows - 3); i < fcRows; i++) {
            System.out.printf("     FC[%d][0-2]: [%.6f, %.6f, %.6f]%n",
                i, initialFCWeights[i][0], initialFCWeights[i][1], initialFCWeights[i][2]);
        }

        // ═══════════════════════════════════════════════════════════
        // ЧАСТЬ 2: Обучаем на 10 примерах
        // ═══════════════════════════════════════════════════════════
        System.out.println("\n📚 PART 2: Training on 10 examples...");

        for (int i = 0; i < testSet.size(); i++) {
            Image img = testSet.get(i);
            List<double[][]> input = new ArrayList<>();
            input.add(img.data());

            double[] output = network.getLayers().getFirst().getOutput(input);
            double[] target = new double[10];
            target[img.label()] = 1.0;

            double[] error = new double[10];
            for (int j = 0; j < 10; j++) {
                error[j] = output[j] - target[j];
            }

            network.getLayers().getLast().backPropagation(error);

            if (i == 0) {
                System.out.printf("   Image 1 - Label: %d, Error magnitude: %.4f%n",
                    img.label(), computeMagnitude(error));
            }
        }

        System.out.println("   ✅ Training completed");

        // ═══════════════════════════════════════════════════════════
        // ЧАСТЬ 3: Анализируем изменения весов
        // ═══════════════════════════════════════════════════════════
        System.out.println("\n🔬 PART 3: Analyzing weight changes...");

        // Conv weights
        System.out.println("\n   📈 Convolutional Layer:");
        int convChanged = 0;
        int convUnchanged = 0;
        double convMaxDelta = 0;
        double convTotalDelta = 0;

        for (int f = 0; f < 8; f++) {
            double[][] filter = convLayer.getFilters().get(f);
            int idx = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double oldVal = initialConvWeights[f][idx++];
                    double newVal = filter[i][j];
                    double delta = Math.abs(newVal - oldVal);

                    convTotalDelta += delta;
                    if (delta > 1e-10) {
                        convChanged++;
                    } else {
                        convUnchanged++;
                    }

                    if (delta > convMaxDelta) {
                        convMaxDelta = delta;
                    }
                }
            }
        }

        System.out.printf("      Changed weights: %d / 72 (%.1f%%)%n",
            convChanged, (convChanged * 100.0) / 72);
        System.out.printf("      Unchanged weights: %d / 72 (%.1f%%)%n",
            convUnchanged, (convUnchanged * 100.0) / 72);
        System.out.printf("      Max delta: %.8f%n", convMaxDelta);
        System.out.printf("      Avg delta: %.8f%n", convTotalDelta / 72);

        // FC weights - анализируем по зонам
        System.out.println("\n   📈 Fully Connected Layer:");

        // Общая статистика
        int fcChanged = 0;
        int fcUnchanged = 0;
        double fcMaxDelta = 0;
        double fcTotalDelta = 0;
        int fcTotalWeights = fcRows * fcCols;

        for (int i = 0; i < fcRows; i++) {
            for (int j = 0; j < fcCols; j++) {
                double oldVal = initialFCWeights[i][j];
                double newVal = fcLayer.getWeights()[i][j];
                double delta = Math.abs(newVal - oldVal);

                fcTotalDelta += delta;
                if (delta > 1e-10) {
                    fcChanged++;
                } else {
                    fcUnchanged++;
                }

                if (delta > fcMaxDelta) {
                    fcMaxDelta = delta;
                }
            }
        }

        System.out.printf("      Total changed: %d / %d (%.1f%%)%n",
            fcChanged, fcTotalWeights, (fcChanged * 100.0) / fcTotalWeights);
        System.out.printf("      Total unchanged: %d / %d (%.1f%%)%n",
            fcUnchanged, fcTotalWeights, (fcUnchanged * 100.0) / fcTotalWeights);
        System.out.printf("      Max delta: %.8f%n", fcMaxDelta);
        System.out.printf("      Avg delta: %.8f%n", fcTotalDelta / fcTotalWeights);

        // Анализ по зонам (первая треть, средняя треть, последняя треть)
        System.out.println("\n      Analysis by input regions:");

        int thirdSize = fcRows / 3;
        String[] regions = {"First third (often zeros from ReLU)", "Middle third", "Last third (often non-zeros)"};
        int[] starts = {0, thirdSize, 2 * thirdSize};
        int[] ends = {thirdSize, 2 * thirdSize, fcRows};

        for (int region = 0; region < 3; region++) {
            int regionChanged = 0;
            int regionTotal = 0;
            double regionMaxDelta = 0;

            for (int i = starts[region]; i < ends[region]; i++) {
                for (int j = 0; j < fcCols; j++) {
                    double oldVal = initialFCWeights[i][j];
                    double newVal = fcLayer.getWeights()[i][j];
                    double delta = Math.abs(newVal - oldVal);

                    regionTotal++;
                    if (delta > 1e-10) {
                        regionChanged++;
                    }
                    if (delta > regionMaxDelta) {
                        regionMaxDelta = delta;
                    }
                }
            }

            System.out.printf("        %s: %d / %d changed (%.1f%%), max Δ=%.6f%n",
                regions[region], regionChanged, regionTotal,
                (regionChanged * 100.0) / regionTotal, regionMaxDelta);
        }

        // Примеры изменений из разных частей
        System.out.println("\n      Sample FC weight changes (first rows):");
        for (int i = 0; i < Math.min(3, fcRows); i++) {
            for (int j = 0; j < Math.min(3, fcCols); j++) {
                double oldVal = initialFCWeights[i][j];
                double newVal = fcLayer.getWeights()[i][j];
                System.out.printf("        FC[%d][%d]: %.6f → %.6f (Δ=%.6f)%n",
                    i, j, oldVal, newVal, newVal - oldVal);
            }
        }

        System.out.println("\n      Sample FC weight changes (last rows):");
        for (int i = Math.max(0, fcRows - 3); i < fcRows; i++) {
            for (int j = 0; j < Math.min(3, fcCols); j++) {
                double oldVal = initialFCWeights[i][j];
                double newVal = fcLayer.getWeights()[i][j];
                System.out.printf("        FC[%d][%d]: %.6f → %.6f (Δ=%.6f)%n",
                    i, j, oldVal, newVal, newVal - oldVal);
            }
        }

        // ═══════════════════════════════════════════════════════════
        // ЧАСТЬ 4: Проверка bias
        // ═══════════════════════════════════════════════════════════
        System.out.println("\n   📈 FC Biases:");
        double[] biases = fcLayer.getBiases();
        System.out.print("      Sample biases: [");
        for (int i = 0; i < Math.min(10, biases.length); i++) {
            System.out.printf("%.4f", biases[i]);
            if (i < Math.min(10, biases.length) - 1) System.out.print(", ");
        }
        System.out.println("]");

        // ═══════════════════════════════════════════════════════════
        // ИТОГОВЫЙ ВЕРДИКТ
        // ═══════════════════════════════════════════════════════════
        System.out.println("\n" + "═".repeat(60));
        System.out.println("📋 VERDICT:");

        if (convChanged > 0) {
            System.out.println("   ✅ Convolutional layer: UPDATING");
        } else {
            System.out.println("   ❌ Convolutional layer: NOT UPDATING");
        }

        if (fcChanged > 0) {
            double percentChanged = (fcChanged * 100.0) / fcTotalWeights;
            System.out.printf("   ✅ Fully Connected layer: UPDATING (%.1f%% of weights changed)%n", percentChanged);

            if (percentChanged < 10) {
                System.out.println("      ⚠️ Note: Low update percentage is EXPECTED with ReLU activation");
                System.out.println("      ℹ️ ReLU creates sparse activations (~70-80% zeros)");
                System.out.println("      ℹ️ Only weights for non-zero inputs update");
            }
        } else {
            System.out.println("   ❌ Fully Connected layer: NOT UPDATING");
        }

        if (convChanged > 0 && fcChanged > 0) {
            System.out.println("\n   🎉 Both layers are updating correctly!");
            System.out.println("   ℹ️ The network is working as expected.");
        } else if (convChanged > 0) {
            System.out.println("\n   ⚠️ Only Conv layer updating - FC layer has a problem!");
        } else {
            System.out.println("\n   ❌ Neither layer updating - major problem!");
        }
    }

    private static double computeMagnitude(double[] vector) {
        double sum = 0;
        for (double v : vector) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }
}
