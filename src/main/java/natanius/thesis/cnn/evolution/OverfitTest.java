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

public class OverfitTest {

    public static void testOverfitting(List<Image> allImages) {
        System.out.println("🧪 OVERFIT TEST: Training on 5 examples");
        System.out.println("Expected: 100% accuracy within 500 epochs");
        System.out.println("─".repeat(60));

        List<Image> tinyDataset = allImages.subList(0, 5);

        System.out.print("Classes: ");
        for (Image img : tinyDataset) {
            System.out.print(img.label() + " ");
        }
        System.out.println("\n");

        NeuralNetwork network = createSimpleNetwork();
        System.out.println("Network architecture:\n" + network);
        System.out.println();

        // 🔍 ДИАГНОСТИКА 1: Проверяем входные данные
        System.out.println("🔍 DIAGNOSTIC: Checking input data...");
        Image firstImage = tinyDataset.get(0);
        double[][] data = firstImage.data();
        double minVal = Double.MAX_VALUE;
        double maxVal = Double.MIN_VALUE;
        double sum = 0;
        int count = 0;

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                double val = data[i][j];
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
                sum += val;
                count++;
            }
        }

        System.out.printf("   Input range: [%.6f, %.6f]%n", minVal, maxVal);
        System.out.printf("   Input mean: %.6f%n", sum / count);

        if (maxVal > 10) {
            System.out.println("   ⚠️ WARNING: Input not normalized! Should be [0, 1]");
        } else {
            System.out.println("   ✅ Input normalized correctly");
        }
        System.out.println();

        // 🔍 ДИАГНОСТИКА 2: Сохраняем начальные веса
        System.out.println("🔍 DIAGNOSTIC: Saving initial weights...");
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

        double initialConvWeight = 0;
        double initialFCWeight = 0;

        if (convLayer != null) {
            initialConvWeight = convLayer.getFilters().get(0)[0][0];
            System.out.printf("   Initial Conv weight[0][0][0]: %.8f%n", initialConvWeight);
        }

        if (fcLayer != null) {
            initialFCWeight = fcLayer.getWeights()[0][0];
            System.out.printf("   Initial FC weight[0][0]: %.8f%n", initialFCWeight);
        }
        System.out.println();

        // 🔍 ДИАГНОСТИКА 3: Проверяем первый forward pass
        System.out.println("🔍 DIAGNOSTIC: Testing forward pass...");
        List<double[][]> input = new ArrayList<>();
        input.add(firstImage.data());
        double[] output = network.getLayers().getFirst().getOutput(input);

        System.out.print("   Output values: [");
        for (int i = 0; i < Math.min(10, output.length); i++) {
            System.out.printf("%.4f", output[i]);
            if (i < output.length - 1) System.out.print(", ");
        }
        System.out.println("]");

        double outputSum = 0;
        for (double v : output) {
            outputSum += v;
        }
        System.out.printf("   Output sum: %.6f%n", outputSum);
        System.out.println();

        // Обучаем с детальной диагностикой
        for (int epoch = 1; epoch <= 500; epoch++) {
            for (Image img : tinyDataset) {
                List<double[][]> inp = new ArrayList<>();
                inp.add(img.data());

                double[] out = network.getLayers().getFirst().getOutput(inp);
                double[] target = new double[10];
                target[img.label()] = 1.0;

                double[] error = new double[10];
                for (int i = 0; i < 10; i++) {
                    error[i] = out[i] - target[i];
                }

                // 🔍 ДИАГНОСТИКА 4: Проверяем градиенты на первой эпохе
                if (epoch == 1) {
                    System.out.println("🔍 DIAGNOSTIC: Checking gradients (epoch 1)...");
                    System.out.print("   Error values: [");
                    for (int i = 0; i < error.length; i++) {
                        System.out.printf("%.4f", error[i]);
                        if (i < error.length - 1) System.out.print(", ");
                    }
                    System.out.println("]");

                    double errorMagnitude = 0;
                    for (double e : error) {
                        errorMagnitude += e * e;
                    }
                    System.out.printf("   Error magnitude: %.6f%n", Math.sqrt(errorMagnitude));
                    System.out.println();
                }

                network.getLayers().getLast().backPropagation(error);
            }

            // Проверяем обновление весов после первой эпохи
            if (epoch == 1) {
                System.out.println("🔍 DIAGNOSTIC: Checking weight updates after epoch 1...");

                if (convLayer != null) {
                    double newConvWeight = convLayer.getFilters().get(0)[0][0];
                    double convDelta = newConvWeight - initialConvWeight;
                    System.out.printf("   Conv weight[0][0][0]: %.8f (delta: %.8f)%n",
                        newConvWeight, convDelta);

                    if (Math.abs(convDelta) < 1e-10) {
                        System.out.println("   ❌ Conv weights NOT updating!");
                    } else {
                        System.out.println("   ✅ Conv weights updating");
                    }
                }

                if (fcLayer != null) {
                    double newFCWeight = fcLayer.getWeights()[0][0];
                    double fcDelta = newFCWeight - initialFCWeight;
                    System.out.printf("   FC weight[0][0]: %.8f (delta: %.8f)%n",
                        newFCWeight, fcDelta);

                    if (Math.abs(fcDelta) < 1e-10) {
                        System.out.println("   ❌ FC weights NOT updating!");
                    } else {
                        System.out.println("   ✅ FC weights updating");
                    }
                }
                System.out.println();
            }

            if (epoch % 50 == 0) {
                int correct = 0;
                double totalLoss = 0.0;

                for (Image img : tinyDataset) {
                    List<double[][]> inp = new ArrayList<>();
                    inp.add(img.data());
                    double[] out = network.getLayers().getFirst().getOutput(inp);

                    int predicted = argMax(out);
                    if (predicted == img.label()) {
                        correct++;
                    }

                    double[] target = new double[10];
                    target[img.label()] = 1.0;
                    totalLoss += computeLoss(out, target);
                }

                double accuracy = (correct * 100.0) / tinyDataset.size();
                double avgLoss = totalLoss / tinyDataset.size();

                System.out.printf("Epoch %3d: Accuracy = %.1f%% (%d/5 correct) | Loss = %.4f%n",
                    epoch, accuracy, correct, avgLoss);

                if (accuracy == 100.0) {
                    System.out.println("✅ SUCCESS: 100% accuracy reached at epoch " + epoch);
                    return;
                }
            }
        }

        System.out.println("❌ FAILED: Could not reach 100% accuracy in 500 epochs");
        System.out.println("This indicates a bug in forward/backward pass");
    }

    private static NeuralNetwork createSimpleNetwork() {
        return new NetworkBuilder()
            .addConvolutionLayer(8, 3, 1, 0.01, new ReLU(), 1)
            .addMaxPoolLayer(2, 2)
            .addFullyConnectedLayer(0.01, new Linear())
            .build();
    }

    private static int argMax(double[] array) {
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static double computeLoss(double[] output, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < output.length; i++) {
            loss -= target[i] * Math.log(Math.max(output[i], 1e-10));
        }
        return loss;
    }
}
