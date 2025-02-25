package natanius.thesis.cnn.evolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static java.util.Collections.shuffle;
import static java.util.stream.Collectors.toList;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.data.DataReader;
import natanius.thesis.cnn.evolution.data.ExcelLogger;
import natanius.thesis.cnn.evolution.data.Image;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.network.ModelRecord;
import natanius.thesis.cnn.evolution.network.NetworkBuilder;
import natanius.thesis.cnn.evolution.network.NeuralNetwork;
import natanius.thesis.cnn.evolution.visualization.FormDigits;

public class Main {

    private static final long SEED = 123;
    private static final int EPOCHS = 100;
    private static final boolean FAST_MODE = true;
    private static final boolean CREATE_NEW_NETWORK = true;
    private static List<Image> IMAGES_TEST;
    private static List<Image> IMAGES_TRAIN;
    private static final List<ModelRecord> MODEL_RECORDS = new ArrayList<>();



    public static void main(String[] args) throws IOException, ClassNotFoundException {

        loadData();

        NeuralNetwork neuralNetwork;
        if (CREATE_NEW_NETWORK) {
            NetworkBuilder builder = new NetworkBuilder(28, 28, 25600)
                .addConvolutionLayer(8, 5, 1, 0.5, SEED)
                .addMaxPoolLayer(3, 2)
                .addFullyConnectedLayer(10, 0.3, SEED);
            neuralNetwork = builder.build();
        } else {
            neuralNetwork = initialize();
        }
        System.out.println(neuralNetwork);


        float rate = neuralNetwork.test(IMAGES_TEST);
        System.out.println("Pre training success rate: " + rate);

        for (int i = 1; i <= EPOCHS; i++) {
            conductExperiment(neuralNetwork, i);

        }

        FormDigits f = new FormDigits(neuralNetwork);
        new Thread(f).start();
    }

    private static void conductExperiment(NeuralNetwork neuralNetwork, int i) {
        long start = now().getEpochSecond();
        shuffle(IMAGES_TRAIN);
        neuralNetwork.train(IMAGES_TRAIN);
        long trainingTime = now().getEpochSecond() - start;

        float trainAccuracy = neuralNetwork.test(IMAGES_TRAIN);
        float testAccuracy = neuralNetwork.test(IMAGES_TEST);
        int totalParams = neuralNetwork.getLayers().stream()
            .map(Layer::getParameterCount)
            .reduce(Integer::sum).orElseThrow();

        System.out.println("Success rate after epoch " + i + ": " + testAccuracy);
        printTimeTaken(trainingTime);

        String modelName = generateModelFileName(2, i);
        saveToFile(neuralNetwork, modelName);
        ExcelLogger.saveResults(modelName, i, testAccuracy, trainAccuracy, totalParams, trainingTime);
        String fullFileName = "models/" + modelName + ".ser";
        MODEL_RECORDS.add(new ModelRecord(testAccuracy, fullFileName));

        if(i % 5 == 0) {
            System.out.println(MODEL_RECORDS);
            cleanupModelFilesForBlock();
        }
    }

    private static void cleanupModelFilesForBlock() {


        List<ModelRecord> toKeep = MODEL_RECORDS.stream().sorted((a, b) -> Float.compare(b.testAccuracy(), a.testAccuracy())).limit(2).toList();

        List<ModelRecord> toDelete = new ArrayList<>();
        for(ModelRecord modelRecord : MODEL_RECORDS) {
            if(!toKeep.contains(modelRecord)) {
                File f = new File(modelRecord.fileName());
                toDelete.add(modelRecord);
                if(f.exists() && f.delete()) {
                    System.out.println("Deleted file: " + modelRecord.fileName());
                } else {
                    System.out.println("Failed to delete file: " + modelRecord.fileName());
                }
            }
        }
        MODEL_RECORDS.removeAll(toDelete);
    }


    private static void loadData() {
        System.out.println("Starting data loading...");

        IMAGES_TEST = new DataReader().readData("data/mnist_test.csv");
        IMAGES_TRAIN = new DataReader().readData("data/mnist_train.csv");

        if (FAST_MODE) {
            IMAGES_TRAIN = IMAGES_TRAIN.stream()
                .limit(100)
                .collect(toList());
        }

        System.out.println("Images Train size: " + IMAGES_TRAIN.size());
        System.out.println("Images Test size: " + IMAGES_TEST.size());
    }

    private static void printTimeTaken(long totalSeconds) {
        long minutes = floorDiv(totalSeconds, 60);
        long seconds = totalSeconds - minutes * 60;
        System.out.printf("Time: %d:%d%n", minutes, seconds);
    }

    private static void saveToFile(NeuralNetwork neuralNetwork, String fileName) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("models/%s.ser".formatted(fileName)))) {
            oos.writeObject(neuralNetwork);
            System.out.println("Neural network saved to file!\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static NeuralNetwork initialize() throws IOException, ClassNotFoundException {
        NeuralNetwork neuralNetwork;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/model1.ser"))) {
            neuralNetwork = (NeuralNetwork) ois.readObject();
            neuralNetwork.linkLayers();
        }
        return neuralNetwork;
    }

    public static String generateModelFileName(int architectureId, int epoch) {
        return String.format("Arch_%d__Epoch_%d", architectureId, epoch);
    }
}
