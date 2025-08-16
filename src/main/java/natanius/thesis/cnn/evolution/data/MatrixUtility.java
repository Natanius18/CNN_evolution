package natanius.thesis.cnn.evolution.data;

import lombok.experimental.UtilityClass;

@UtilityClass
public class MatrixUtility {

    public static void add(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            double[] aRow = a[i];
            double[] bRow = b[i];
            for (int j = 0; j < aRow.length; j++) {
                aRow[j] += bRow[j];
            }
        }
    }

    public static void add(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static double[][] multiplyWithNewArray(double[][] a, double scalar) {
        double[][] out = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[i][j] = a[i][j] * scalar;
            }
        }
        return out;
    }

    public static void multiply(double[][] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                a[i][j] *= scalar;
            }
        }
    }

    public static void multiply(double[] a, double scalar) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= scalar;
        }
    }
}
