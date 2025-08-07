package natanius.thesis.cnn.evolution.network;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static java.util.Collections.shuffle;
import static java.util.stream.Collectors.toList;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTestData;
import static natanius.thesis.cnn.evolution.data.DataReader.loadTrainData;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.data.ExcelLogger;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.layers.Layer;

public class ExperimentalSandbox {
    private List<Image> imagesTrain = loadTrainData();
    private final List<Image> imagesTest = loadTestData();
    private final List<ModelRecord> modelRecords = new ArrayList<>();
    private static final int EPOCHS = 3;
    private static final boolean FAST_MODE = true;


    public ExperimentalSandbox() {
        if (FAST_MODE) {
            imagesTrain = imagesTrain.stream()
                .limit(500)
                .collect(toList());
        }
    }

    public void checkArchitecture(int architectureId, NeuralNetwork neuralNetwork, int[] chromosome) {
        for (int i = 1; i <= EPOCHS; i++) {
            conductExperiment(i, architectureId, neuralNetwork, chromosome);
        }
    }

    private void conductExperiment(int epoch, int architectureId, NeuralNetwork neuralNetwork, int[] chromosome) {
        System.out.printf("===================================================================================== Arch %d epoch %d%n", architectureId, epoch);
        shuffle(imagesTrain);
        long start = now().getEpochSecond();

        neuralNetwork.train(imagesTrain, 3);

        long trainingTime = now().getEpochSecond() - start;
        printTimeTaken(trainingTime);
        saveResults(neuralNetwork, epoch, architectureId, trainingTime, chromosome);
    }

    private void saveResults(NeuralNetwork neuralNetwork, int epoch, int architectureId, long trainingTime, int[] chromosome) {
        float trainAccuracy = neuralNetwork.test(imagesTrain);
        float testAccuracy = neuralNetwork.test(imagesTest);
        int totalParams = neuralNetwork.getLayers().stream()
            .map(Layer::getParameterCount)
            .reduce(Integer::sum).orElseThrow();
        System.out.println("Success rate after epoch " + epoch + ": " + testAccuracy);

        String modelName = generateModelFileName(architectureId, epoch);
        saveToFile(modelName, neuralNetwork);
        ExcelLogger.saveResults(modelName, epoch, testAccuracy, trainAccuracy, totalParams, trainingTime, chromosome);

        modelRecords.add(new ModelRecord(testAccuracy, "models/" + modelName + ".ser"));
        if (epoch % 5 == 0) {
            System.out.println(modelRecords);
            cleanupModelFiles();
        }
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n", minutes, seconds);
    }
    private void saveToFile(String fileName, NeuralNetwork neuralNetwork) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("models/" + fileName + ".ser"))) {
            oos.writeObject(neuralNetwork);
            System.out.println("Model saved: " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void cleanupModelFiles() {

        List<ModelRecord> toKeep = modelRecords.stream()
            .sorted((a, b) -> Float.compare(b.testAccuracy(), a.testAccuracy()))
            .limit(2)
            .toList();

        for(ModelRecord modelRecord : modelRecords) {
            if(!toKeep.contains(modelRecord)) {
                File f = new File(modelRecord.fileName());
                if(f.exists() && f.delete()) {
                    System.out.println("Deleted file: " + modelRecord.fileName());
                } else {
                    System.out.println("Failed to delete file: " + modelRecord.fileName());
                }
            }
        }
        modelRecords.removeIf(model -> !toKeep.contains(model));
    }

    private String generateModelFileName(int architectureId, int epoch) {
        return String.format("Arch_%d__Epoch_%d", architectureId, epoch);
    }
}
