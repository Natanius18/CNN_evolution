package natanius.thesis.cnn.evolution.activation;

import static natanius.thesis.cnn.evolution.data.Constants.LEAK;

public class LeakyReLU implements Activation {

    @Override
    public double forward(double z) {
        return z <= 0 ? LEAK * z : z;
    }

    @Override
    public double backward(double z) {
        return z > 0 ? 1.0 : LEAK;
    }

}
