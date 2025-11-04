package natanius.thesis.cnn.evolution.data;

import static lombok.AccessLevel.PRIVATE;

import java.util.Random;
import lombok.NoArgsConstructor;
import natanius.thesis.cnn.evolution.activation.Activation;
import natanius.thesis.cnn.evolution.activation.LeakyReLU;
import natanius.thesis.cnn.evolution.activation.ReLU;
import natanius.thesis.cnn.evolution.activation.Sigmoid;

@NoArgsConstructor(access = PRIVATE)
public class Constants {
    //    100% constants
    public static final long SEED = 123L;
    public static final Random RANDOM = new Random(SEED);
    public static final int INPUT_ROWS = 28;
    public static final int INPUT_COLS = 28;
    public static final int OUTPUT_CLASSES = 10;
    public static final int[] ALLOWED_FC_SIZES = {64, 128, 256, 512};
    public static final int SCALE_FACTOR = 255;
    public static final double LEAK = 0.01;
    public static final int[] ALLOWED_FILTER_SIZES = {3, 5, 7};
    public static final int[] ALLOWED_FILTERS = {4, 8, 16}; //, 32, 64};
    public static final Activation[] ACTIVATION_STRATEGIES = {new ReLU(), new LeakyReLU(), new Sigmoid()};

    public static final int MIN_CONV_BLOCKS = 1;
    public static final int MAX_CONV_BLOCKS = 4;
    public static final int MAX_FC_LAYERS = 3;  // hidden + output
    public static final int CONV_STEP_SIZE = 1;
    public static double getLearningRate(Activation activation) {
        if (activation instanceof Sigmoid) {
            return 0.01;
        } else {
            return 0.001;
        }
    }

    public static final int MAX_POOL_STEP_SIZE = 2; //todo google what are valid nums
    public static final int MAX_POOL_WINDOW_SIZE = 2;
    public static final double LEARNING_RATE_FULLY_CONNECTED = 0.001;

    // Evolution-related parameters
    public static final int POPULATION_SIZE = 40;
    public static final int GENERATIONS = 5;
    public static final int ELITE_COUNT = (int) (POPULATION_SIZE * 0.1);       // 10%
    public static final int CROSSOVER_COUNT = (int) (POPULATION_SIZE * 0.5);  // 50%
    public static final int MUTANT_COUNT = (int) (POPULATION_SIZE * 0.3);    // 30%
    public static final float DATASET_FRACTION = 0.005f;


    public static final int EPOCHS = 5;
    public static final boolean FAST_MODE = true;
    public static final boolean DEBUG = false;
}
