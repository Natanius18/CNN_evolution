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
     * Обучение нейросети на протяжении numEpochs эпох с mini-batch
     *
     * @param neuralNetwork нейросеть для обучения
     * @param trainSet      тренировочный набор данных
     * @param validationSet набор для валидации
     * @return точность на последней эпохе
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
