package natanius.thesis.cnnEvolution;

import static java.lang.Math.floorDiv;
import static java.time.Instant.now;
import static java.util.Collections.shuffle;

import java.time.Instant;
import java.util.List;
import natanius.thesis.cnnEvolution.data.DataReader;
import natanius.thesis.cnnEvolution.data.Image;
import natanius.thesis.cnnEvolution.network.NetworkBuilder;
import natanius.thesis.cnnEvolution.network.NeuralNetwork;
import natanius.thesis.cnnEvolution.visualization.FormDigits;

public class Main {

    public static void main(String[] args) {

        long SEED = 123;

        System.out.println("Starting data loading...");

        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");
//        imagesTrain = imagesTrain.stream().filter(image -> image.label() == 0).collect(Collectors.toList());

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size: " + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 25600);
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();
        System.out.println(net);

        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 1;

        for (int i = 0; i < epochs; i++) {
            Instant start = now();
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
            long totalSeconds = now().getEpochSecond() - start.getEpochSecond();
            long mins = floorDiv(totalSeconds, 60);
            long seconds = totalSeconds - mins * 60;
            System.out.printf("Time: %d:%d%n", mins, seconds);
        }

        FormDigits f = new FormDigits(net);
        new Thread(f).start();
    }
}
