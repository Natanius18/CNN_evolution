package natanius.thesis.cnn.evolution.data;

import lombok.experimental.UtilityClass;

@UtilityClass
public class MatrixUtility {

    public static void add(double[][] a, double[][] b) {
        int length = a.length;
        for (int i = 0; i < length; i++) {
            double[] aRow = a[i];
            double[] bRow = b[i];
            for (int j = 0; j < length; j++) {
                aRow[j] += bRow[j];
            }
        }
    }

    public static void add(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static void multiply(double[][] a, double scalar) {
        int length = a.length;
        for (double[] row : a) {
            for (int j = 0; j < length; j++) {
                row[j] *= scalar;
            }
        }
    }

    public static void multiply(double[] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= scalar;
        }
    }
}
