package natanius.thesis.cnn.evolution.data;

import static lombok.AccessLevel.PRIVATE;

import java.util.Random;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = PRIVATE)
public class Constants {
    public static final int CONV_LAYERS = 2;
    public static final int POPULATION_SIZE = 10;
    public static final int ELITE_COUNT = (int) (POPULATION_SIZE * 0.1);       // 10%
    public static final int CROSSOVER_COUNT = (int) (POPULATION_SIZE * 0.5);  // 50%
    public static final int MUTANT_COUNT = (int) (POPULATION_SIZE * 0.3);    // 30%
    public static final int GENERATIONS = 10;
    public static final int MAX_FILTERS = 64;
    public static final int MAX_FILTER_SIZE = 7;
    public static final float LEARNING_RATE = 0.3f;
    public static final double LEARNING_RATE_FULLY_CONNECTED = 0.7;
    public static final long SEED = 123L;
    public static final Random RANDOM = new Random(SEED);
    public static final int INPUT_ROWS = 28;
    public static final int INPUT_COLS = 28;
    public static final int OUTPUT_CLASSES = 10;
    public static final int SCALE_FACTOR = 25600;
    public static final int EPOCHS = 3;
    public static final int BATCH_SIZE = 3;
    public static final boolean FAST_MODE = true;
    public static final boolean TRAIN_AND_SAVE_BEST_MODEL = false;
}
