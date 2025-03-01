package natanius.thesis.cnn.evolution.data;

import static lombok.AccessLevel.PRIVATE;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import lombok.NoArgsConstructor;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

@NoArgsConstructor(access = PRIVATE)
public class ExcelLogger {

    private static final String FILE_PATH = "models/training_results.xlsx";

    public static synchronized void saveResults(String modelName,
                                   int epoch,
                                   float testAccuracy,
                                   float trainAccuracy,
                                   int totalParams,
                                   long trainingTime) {
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
        row.createCell(0).setCellValue(modelName);
        row.createCell(1).setCellValue(epoch);
        row.createCell(2).setCellValue(testAccuracy);
        row.createCell(3).setCellValue(trainAccuracy);
        row.createCell(4).setCellValue(totalParams);
        row.createCell(5).setCellValue(trainingTime);

        try (FileOutputStream fos = new FileOutputStream(FILE_PATH)) {
            workbook.write(fos);
            workbook.close();
        } catch (IOException e) {
            throw new RuntimeException("Error writing to Excel file", e);
        }
    }

    private static void createHeader(Sheet sheet) {
        Row header = sheet.createRow(0);
        String[] columns = {"Model Name", "Epoch", "Test Accuracy", "Train Accuracy", "Total Parameters", "Training Time (s)"};
        for (int i = 0; i < columns.length; i++) {
            header.createCell(i).setCellValue(columns[i]);
        }
    }
}

