package natanius.thesis.cnn.evolution.network;

import static java.util.Collections.shuffle;
import static natanius.thesis.cnn.evolution.data.Constants.RANDOM;

import java.util.List;
import lombok.Getter;
import natanius.thesis.cnn.evolution.data.Image;

@Getter
public class EpochTrainer {

    private final int batchSize;
    private final int numEpochs;

    /**
     * @param batchSize размер мини-батча для обучения
     * @param numEpochs кількість епох
     */
    public EpochTrainer(int batchSize, int numEpochs) {
        this.batchSize = batchSize;
        this.numEpochs = numEpochs;
    }

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

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            // Перемешиваем тренировочный набор перед каждой эпохой
            shuffle(trainSet, RANDOM);

            // Обучение на эпохе с mini-batch
            double epochLoss = neuralNetwork.trainEpoch(trainSet, batchSize);

            // Тестирование на validation set
            accuracy = neuralNetwork.testBatch(validationSet, batchSize);

            System.out.printf("Epoch %d/%d | Loss: %.6f | Validation Accuracy: %.4f%n",
                epoch, numEpochs, epochLoss, accuracy);
        }

        return accuracy;
    }

}
