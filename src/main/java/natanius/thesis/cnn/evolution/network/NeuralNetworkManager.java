package natanius.thesis.cnn.evolution.network;

import static lombok.AccessLevel.PRIVATE;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = PRIVATE)
public class NeuralNetworkManager {
    private static final long SEED = 123;
    private static final boolean CREATE_NEW_NETWORK = true;

    public static NeuralNetwork initializeNetwork() throws IOException, ClassNotFoundException {
        if (CREATE_NEW_NETWORK) {
            return new NetworkBuilder(28, 28, 25600)
                .addConvolutionLayer(8, 5, 1, 0.3, SEED)
                .addMaxPoolLayer(3, 2)
                .addFullyConnectedLayer(10, 0.7, SEED)
                .build();
        } else {
            return loadFromFile();
        }
    }

    private static NeuralNetwork loadFromFile() throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/model1.ser"))) {
            NeuralNetwork network = (NeuralNetwork) ois.readObject();
            network.linkLayers();
            return network;
        }
    }
}
