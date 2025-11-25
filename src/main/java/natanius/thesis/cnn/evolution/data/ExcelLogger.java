package natanius.thesis.cnn.evolution.data;

import static lombok.AccessLevel.PRIVATE;
import static natanius.thesis.cnn.evolution.data.Constants.DATASET_FRACTION;
import static natanius.thesis.cnn.evolution.genes.GeneticAlgorithm.CACHE;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import lombok.NoArgsConstructor;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

@NoArgsConstructor(access = PRIVATE)
public class ExcelLogger {

    private static final String FILE_PATH = "logs/training_results.xlsx";
    private static final String CACHE_FILE_PATH = "logs/cache_results.xlsx";

    public static synchronized void saveResults(int generation,
                                                float fitness,
                                                float testAccuracy,
                                                float validationAccuracy,
                                                float trainAccuracy,
                                                int totalParams,
                                                long trainingTime,
                                                String chromosome) {

        File file = new File(FILE_PATH);
        Workbook workbook;
        Sheet sheet;

        if (!file.exists()) {
            workbook = new XSSFWorkbook();
            sheet = workbook.createSheet("Training Results");
            createHeader(sheet);
        } else {
            try (FileInputStream fis = new FileInputStream(FILE_PATH)) {
                workbook = new XSSFWorkbook(fis);
                sheet = workbook.getSheetAt(0);
            } catch (IOException e) {
                throw new RuntimeException("Error reading Excel file", e);
            }
        }

        int rowNum = sheet.getLastRowNum() + 1;
        Row row = sheet.createRow(rowNum);
        row.createCell(0).setCellValue(generation);
        row.createCell(1).setCellValue(fitness);
        row.createCell(2).setCellValue(trainAccuracy);
        row.createCell(3).setCellValue(validationAccuracy);
        row.createCell(4).setCellValue(testAccuracy);
        row.createCell(5).setCellValue(trainAccuracy - testAccuracy);  // Overfitting gap
        row.createCell(6).setCellValue(testAccuracy / totalParams * 1000);  // Parameters efficiency
        row.createCell(7).setCellValue(totalParams);
        row.createCell(8).setCellValue(trainingTime);
        row.createCell(9).setCellValue(chromosome);
        row.createCell(10).setCellValue(LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm")));
        row.createCell(11).setCellValue(DATASET_FRACTION * 100 + "%");

        for (int i = 0; i <= 10; i++) {
            sheet.autoSizeColumn(i);
        }


        try (FileOutputStream fos = new FileOutputStream(FILE_PATH)) {
            workbook.write(fos);
            workbook.close();
        } catch (IOException e) {
            throw new RuntimeException("Error writing to Excel file", e);
        }
    }

    private static void createHeader(Sheet sheet) {
        Row header = sheet.createRow(0);
        String[] columns = {
            "Generation", "Fitness", "Train Accuracy", "Validation Accuracy", "Test Accuracy",
            "Overfitting Gap", "Params Efficiency (×1000)",
            "Total Parameters", "Training Time for generation (s)", "Chromosome", "Date Time", "Dataset Fraction"
        };
        for (int i = 0; i < columns.length; i++) {
            header.createCell(i).setCellValue(columns[i]);
        }
    }

    public static synchronized void saveCacheToExcel() {
        Workbook workbook = new XSSFWorkbook();
        Sheet sheet = workbook.createSheet("Cache");

        Row header = sheet.createRow(0);
        header.createCell(0).setCellValue("Architecture");
        header.createCell(1).setCellValue("Fitness");

        int rowNum = 1;
        for (Map.Entry<String, Float> entry : CACHE.entrySet()) {
            Row row = sheet.createRow(rowNum++);
            row.createCell(0).setCellValue(entry.getKey());
            Float fitness = entry.getValue();
            if (fitness != null) {
                row.createCell(1).setCellValue(fitness);
            }
        }

        sheet.autoSizeColumn(0);
        sheet.autoSizeColumn(1);

        try (FileOutputStream fos = new FileOutputStream(CACHE_FILE_PATH)) {
            workbook.write(fos);
            workbook.close();
            System.out.println("Cache saved to " + CACHE_FILE_PATH);
        } catch (IOException e) {
            throw new RuntimeException("Error writing cache to Excel file", e);
        }
    }

    public static synchronized void saveArchitectureTestResults(String architecture, int epoch, double loss, float trainAccuracy, float testAccuracy, long trainingTime) {
        String filePath = "logs/architecture_test_results.xlsx";
        File file = new File(filePath);
        Workbook workbook;
        Sheet sheet;

        if (!file.exists()) {
            workbook = new XSSFWorkbook();
            sheet = workbook.createSheet("Test Results");
            Row header = sheet.createRow(0);
            header.createCell(0).setCellValue("Architecture");
            header.createCell(1).setCellValue("Epoch");
            header.createCell(2).setCellValue("Loss");
            header.createCell(3).setCellValue("Train Accuracy");
            header.createCell(4).setCellValue("Test Accuracy");
            header.createCell(5).setCellValue("Training Time (s)");
        } else {
            try (FileInputStream fis = new FileInputStream(filePath)) {
                workbook = new XSSFWorkbook(fis);
                sheet = workbook.getSheetAt(0);
            } catch (IOException e) {
                throw new RuntimeException("Error reading Excel file", e);
            }
        }

        int rowNum = sheet.getLastRowNum() + 1;
        Row row = sheet.createRow(rowNum);
        row.createCell(0).setCellValue(architecture);
        row.createCell(1).setCellValue(epoch);
        row.createCell(2).setCellValue(loss);
        row.createCell(3).setCellValue(trainAccuracy);
        row.createCell(4).setCellValue(testAccuracy);
        row.createCell(5).setCellValue(trainingTime);

        for (int i = 0; i <= 4; i++) {
            sheet.autoSizeColumn(i);
        }

        try (FileOutputStream fos = new FileOutputStream(filePath)) {
            workbook.write(fos);
            workbook.close();
        } catch (IOException e) {
            throw new RuntimeException("Error writing to Excel file", e);
        }
    }
}
