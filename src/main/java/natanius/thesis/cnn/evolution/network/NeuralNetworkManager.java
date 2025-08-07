//package natanius.thesis.cnn.evolution.network;
//
//import static lombok.AccessLevel.PRIVATE;
//import static natanius.thesis.cnn.evolution.data.Constants.CREATE_NEW_NETWORK;
//
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import lombok.NoArgsConstructor;
//
//@NoArgsConstructor(access = PRIVATE)
//public class NeuralNetworkManager {
//
//
//    public static NeuralNetwork initializeNetwork() throws IOException, ClassNotFoundException {
//        if (CREATE_NEW_NETWORK) {
//            return new NetworkBuilder()
//                .addConvolutionLayer(8, 5, 1, 0.3)
//                .addMaxPoolLayer(3, 2)
//                .addFullyConnectedLayer(10, 0.7)
//                .build();
//        } else {
//            return loadFromFile();
//        }
//    }
//
//    private static NeuralNetwork loadFromFile() throws IOException, ClassNotFoundException {
//        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/model1.ser"))) {
//            NeuralNetwork network = (NeuralNetwork) ois.readObject();
//            network.linkLayers();
//            return network;
//        }
//    }
//}
