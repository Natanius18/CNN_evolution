//package natanius.thesis.cnn.evolution;
//
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.List;
//import natanius.thesis.cnn.evolution.network.ExperimentalSandbox;
//import natanius.thesis.cnn.evolution.network.NeuralNetwork;
//import natanius.thesis.cnn.evolution.network.NeuralNetworkManager;
//
//public class Main {
//
//    private static final ExperimentalSandbox sandbox = new ExperimentalSandbox();
//
//    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
//        int[] architectureIds = {1};
//        List<Thread> threads = new ArrayList<>();
//
//        long startTime = System.currentTimeMillis();
//        for (int archId : architectureIds) {
//            NeuralNetwork neuralNetwork = NeuralNetworkManager.initializeNetwork();
//
//            Thread thread = new Thread(() -> {
//                sandbox.checkArchitecture(archId, neuralNetwork);
//            });
//
//            threads.add(thread);
//            thread.start();
//        }
//
//        for (Thread thread : threads) {
//            try {
//                thread.join();
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//                throw e;
//            }
//        }
//        long endTime = System.currentTimeMillis();
//        System.out.println("Total execution time: " + (endTime - startTime));
//
////        new Thread(new FormDigits(neuralNetwork)).start();
//    }
//}
//
//
//
//
