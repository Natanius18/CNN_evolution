package natanius.thesis.cnn.evolution.activation;

public class ReLU implements Activation {

    @Override
    public double forward(double z) {
        return Math.max(0, z);
    }

    @Override
    public double backward(double z) {
        return z > 0 ? 1.0 : 0.0;
    }

}
