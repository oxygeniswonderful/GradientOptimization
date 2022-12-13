public class Matrix2D implements Matrix{

    @Override
    public double determinant(Double[][] matrix) {
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
    }

    @Override
    public Double[][] reverseMatrix(Double[][] matrix) {
        Double[][] reverseMatrix = new Double[2][2];
        double determinant = determinant(matrix);
        reverseMatrix[0][0] = matrix[1][1] / determinant;
        reverseMatrix[0][1] = -matrix[0][1] / determinant;
        reverseMatrix[1][0] = -matrix[1][0] / determinant;
        reverseMatrix[1][1] = matrix[0][0] / determinant;

        return reverseMatrix;
    }

    @Override
    public Double[][] matrixMultiplication(Double[][] matrix1, Double[][] matrix2) {
        Double[][] result = new Double[2][2];

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                    System.out.println(result[i][j]);
                }
            }
        }
        return result;
    }
}
