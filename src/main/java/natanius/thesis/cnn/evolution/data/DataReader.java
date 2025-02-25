package natanius.thesis.cnn.evolution.data;

import static java.lang.Integer.parseInt;
import static java.util.stream.Collectors.toList;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DataReader {
    public List<Image> readData(String path) {
        try (Stream<String> lines = Files.lines(Paths.get(path))) {
            return lines.parallel()
                .map(this::parseLine)
                .collect(toList());
        } catch (Exception e) {
            throw new RuntimeException("File not found " + path, e);
        }
    }

    private Image parseLine(String line) {
        String[] items = line.split(",");
        int label = parseInt(items[0]);
        double[][] data = new double[28][28];

        IntStream.range(1, items.length)
            .parallel()
            .forEach(i -> data[(i - 1) / 28][(i - 1) % 28] = parseInt(items[i]));

        return new Image(data, label);
    }
}
