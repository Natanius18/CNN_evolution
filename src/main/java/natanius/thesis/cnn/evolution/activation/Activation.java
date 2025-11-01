package natanius.thesis.cnn.evolution.activation;

public interface Activation {
    double forward(double z);
    double backward(double z);
}
