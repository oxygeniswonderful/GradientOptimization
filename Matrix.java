public interface Matrix {
    /**
     * Determinant
     */
    double determinant(Double[][] matrix);

    /**
     * reverse matrix
     */
    Double[][] reverseMatrix(Double[][] matrix);

    /**
     * Matrix multiplication
     */
    Double[][] matrixMultiplication(Double[][] matrix1, Double[][] matrix2);
}
