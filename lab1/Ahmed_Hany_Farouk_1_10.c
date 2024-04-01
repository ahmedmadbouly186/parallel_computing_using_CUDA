/**
    Ahmed_Hany_Farouk_1_10

    @author: Ahmed Hany Farouk
    Sec:    1
    BN:     10
    Code:   9202213

*/

#include <stdio.h>
#include <stdlib.h>

/**
     * @brief Concatenates two numbers by multiplying the first number by 10 raised to the number of digits in the second number and then adding the second number
     * @param a : The first number
     * @param b : The second number
     * @return The concatenated number
*/
long long concat2Numbers(long long a, long long b) {
    long long c = b;
    // Count the number of digits in the second number
    do {
        a *= 10;
        c /= 10;
    } while (c > 0);
    // Return the concatenated number
    return a + b;
}

/**
    * @brief This program takes a matrix and returns the sum of the numbers formed by concatenation every single column of the matrix into a single number
    * @param matrix : The matrix to concatenate
    * @param n_rows : The number of rows to concatenate
    * @param n_cols : The number of columns to concatenate
*/
long long sumOfConcatenatedMatrixNumbers(int ** matrix, int n_rows, int n_cols) {
    // Initialize the sum
    long long sum = 0;
    // Iterate over the columns
    for (int j = 0; j < n_cols; j++) {
        // Initialize the number formed by concatenating the column
        long long number = 0;
        // Iterate over the rows
        for (int i = 0; i < n_rows; i++) {
            // Add the number to the column
            number = concat2Numbers(number, matrix[i][j]);
        }
        // Add the number to the sum
        sum += number;
    }
    // Return the sum
    return sum;
}

/*
                    nrows ncols nrows*ncols numbers
run it like ./program 3     3   9 10 20 30 5 10 20 2 4 6

*/
int main(int argc, char *argv[]) {
    // Initialize the matrix
    int ** matrix;
    // Initialize the number of rows and columns
    int n_rows, n_cols;
    // Check if the number of arguments is correct
    if (argc < 4) {
        printf("Invalid number of arguments\n");
        return 1;
    }
    // Get the number of rows and columns from the arguments
    n_rows = atoi(argv[1]);
    n_cols = atoi(argv[2]);
    // Create a dynamic array to store the matrix
    matrix = (int **) malloc(n_rows * sizeof(int *));
    for (int i = 0; i < n_rows; i++) {
        matrix[i] = (int *) malloc(n_cols * sizeof(int));
    }
    // Get the matrix from the arguments
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            matrix[i][j] = atoi(argv[3 + i * n_cols + j]);
        }
    }
    // Print the sum of the numbers formed by concatenation every single column of the matrix
    printf("%lld\n", sumOfConcatenatedMatrixNumbers(matrix, n_rows, n_cols));
    // Free the memory
    for (int i = 0; i < n_rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
    return 0;
}