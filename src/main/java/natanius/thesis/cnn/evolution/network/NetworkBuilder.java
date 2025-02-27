package natanius.thesis.cnn.evolution.network;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.layers.ConvolutionLayer;
import natanius.thesis.cnn.evolution.layers.FullyConnectedLayer;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.layers.MaxPoolLayer;

public class NetworkBuilder {

    private final List<Layer> layers = new ArrayList<>();
    private final int inputRows;
    private final int inputCols;
    private final int scaleFactor;

    public NetworkBuilder(int inputRows, int inputCols, int scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
    }

    public NetworkBuilder addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed) {
        if (layers.isEmpty()) {
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inputRows, inputCols, seed, numFilters, learningRate));
        } else {
            Layer prev = layers.getLast();
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), seed, numFilters, learningRate));
        }
        return this;
    }

    public NetworkBuilder addMaxPoolLayer(int windowSize, int stepSize) {
        if (layers.isEmpty()) {
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        } else {
            Layer prev = layers.getLast();
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
        return this;
    }

    public NetworkBuilder addFullyConnectedLayer(int outLength, double learningRate, long seed) {
        if (layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(inputCols * inputRows, outLength, seed, learningRate));
        } else {
            Layer prev = layers.getLast();
            layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, seed, learningRate));
        }
        return this;
    }

    public NeuralNetwork build() {
        return new NeuralNetwork(layers, scaleFactor);
    }

}
