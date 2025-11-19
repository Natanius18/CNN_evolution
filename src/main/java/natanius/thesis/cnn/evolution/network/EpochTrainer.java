package natanius.thesis.cnn.evolution.network;

import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.BATCH_SIZE;
import static natanius.thesis.cnn.evolution.data.Constants.EPOCHS;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.List;
import lombok.Getter;
import natanius.thesis.cnn.evolution.data.Image;

@Getter
public class EpochTrainer {

    /**
     * Навчання нейромережі протягом numEpochs епох з mini-batch
     *
     * @param neuralNetwork нейромережа для навчання
     * @param trainSet      тренувальний набір даних
     * @param validationSet набір для валідації
     * @return точність на останній епосі
     */
    public float train(NeuralNetwork neuralNetwork, List<Image> trainSet, List<Image> validationSet) {
        float accuracy = 0;

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            shuffle(trainSet, RANDOM);

            neuralNetwork.trainEpoch(trainSet, BATCH_SIZE);

            accuracy = neuralNetwork.testBatch(validationSet, BATCH_SIZE);
        }

        return accuracy;
    }

}
