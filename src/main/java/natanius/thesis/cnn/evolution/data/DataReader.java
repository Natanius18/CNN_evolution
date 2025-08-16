package natanius.thesis.cnn.evolution.data;

import static java.lang.Integer.parseInt;
import static java.util.stream.Collectors.toList;
import static lombok.AccessLevel.PRIVATE;
import static natanius.thesis.cnn.evolution.data.Constants.SCALE_FACTOR;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = PRIVATE)
public class DataReader {

    public static List<Image> loadTrainData() {
        System.out.println("Loading train data...");
        return readData("data/mnist_train.csv");
    }

    public static List<Image> loadTestData() {
        System.out.println("Loading test data...");
        return readData("data/mnist_test.csv");
    }
    public static List<Image> readData(String path) {
        try (Stream<String> lines = Files.lines(Paths.get(path))) {
            return lines.parallel()
                .map(DataReader::parseLine)
                .collect(toList());
        } catch (Exception e) {
            throw new RuntimeException("File not found " + path, e);
        }
    }

    private static Image parseLine(String line) {
        String[] items = line.split(",");
        int label = parseInt(items[0]);
        double[][] data = new double[28][28];

        IntStream.range(1, items.length)
            .forEach(i -> data[(i - 1) / 28][(i - 1) % 28] = (double) parseInt(items[i]) / SCALE_FACTOR); // нормализация данных

        return new Image(data, label);
    }
}
