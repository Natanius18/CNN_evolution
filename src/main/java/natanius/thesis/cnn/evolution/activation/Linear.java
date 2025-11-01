package natanius.thesis.cnn.evolution.activation;

public class Linear implements Activation {

    @Override
    public double forward(double x) {
        return x;  // f(x) = x
    }

    @Override
    public double backward(double x) {
        return 1.0;  // f'(x) = 1
    }

}

