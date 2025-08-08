package natanius.thesis.cnn.evolution.network;

import static natanius.thesis.cnn.evolution.data.Constants.INPUT_COLS;
import static natanius.thesis.cnn.evolution.data.Constants.INPUT_ROWS;

import java.util.ArrayList;
import java.util.List;
import natanius.thesis.cnn.evolution.layers.ConvolutionLayer;
import natanius.thesis.cnn.evolution.layers.FullyConnectedLayer;
import natanius.thesis.cnn.evolution.layers.Layer;
import natanius.thesis.cnn.evolution.layers.MaxPoolLayer;

public class NetworkBuilder {

    private final List<Layer> layers = new ArrayList<>();


    public NetworkBuilder addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate) {
        if (layers.isEmpty()) {
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, INPUT_ROWS, INPUT_COLS, numFilters, learningRate));
        } else {
            Layer prev = layers.getLast();
            if (prev.getOutputRows() < filterSize || prev.getOutputCols() < filterSize) {
                throw new IllegalStateException("Output too small for filter size " + filterSize);
            }
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), numFilters, learningRate));
        }
        return this;
    }

    public NetworkBuilder addMaxPoolLayer(int windowSize, int stepSize) {
        if (layers.isEmpty()) {
            if (INPUT_COLS < windowSize) {
                throw new IllegalStateException("Input too small for pooling window " + windowSize);
            }
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, INPUT_ROWS, INPUT_COLS));
        } else {
            Layer prev = layers.getLast();
            if (prev.getOutputRows() < windowSize || prev.getOutputCols() < windowSize) {
                throw new IllegalStateException("Output too small for pooling window " + windowSize);
            }
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
        return this;
    }


    public NetworkBuilder addFullyConnectedLayer(double learningRate) {
        if (layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(INPUT_COLS * INPUT_ROWS, learningRate));
        } else {
            Layer prev = layers.getLast();
            int inputElements = prev.getOutputElements();
            if (inputElements <= 0) {
                throw new IllegalStateException("Cannot add fully connected layer: previous layer output is invalid");
            }
            layers.add(new FullyConnectedLayer(inputElements, learningRate));
        }

        return this;
    }


    public NeuralNetwork build() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Cannot build network: no layers added.");
        }

        for (Layer layer : layers) {
            if (layer.getOutputElements() <= 0) {
                throw new IllegalStateException("Layer has invalid output size: " + layer);
            }
        }
        return new NeuralNetwork(layers);
    }

}
