package natanius.thesis.cnn.evolution.network;

import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.BATCH_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.DEBUG;
import static natanius.thesis.cnn.evolution.data.Constants.EPOCHS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.List;
import lombok.Getter;
import natanius.thesis.cnn.evolution.data.Image;

@Getter
public class EpochTrainer {

    public float train(NeuralNetwork neuralNetwork, List<Image> imagesTrain, List<Image> imagesTest) {
        float accuracy = 0;
        for (int i = 1; i <= EPOCHS; i++) {
            if (DEBUG) {
                System.out.printf("===================================================================================== Epoch %d%n", i);
            }
            shuffle(imagesTrain, RANDOM);

            neuralNetwork.train(imagesTrain, BATCH_SIZE);

            accuracy = neuralNetwork.test(imagesTest);
            if (DEBUG) {
                System.out.println("Accuracy: " + accuracy);
            }
        }
        return accuracy;
    }

}
