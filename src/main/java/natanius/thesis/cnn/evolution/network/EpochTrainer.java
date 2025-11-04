package natanius.thesis.cnn.evolution.network;

import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.DEBUG;
import static natanius.thesis.cnn.evolution.data.Constants.EPOCHS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.List;
import lombok.Getter;
import natanius.thesis.cnn.evolution.data.Image;

@Getter
public class EpochTrainer {

    public float train(NeuralNetwork neuralNetwork, List<Image> trainSet, List<Image> validationSet) {
        float accuracy = 0;
        for (int i = 1; i <= EPOCHS; i++) {
            if (DEBUG) {
                System.out.printf("===================================================================================== Epoch %d%n", i);
            }
            shuffle(trainSet, RANDOM);

            neuralNetwork.train(trainSet);

            accuracy = neuralNetwork.test(validationSet);
            if (DEBUG) {
                System.out.println("Accuracy: " + accuracy);
            }
        }
        return accuracy;
    }

}
