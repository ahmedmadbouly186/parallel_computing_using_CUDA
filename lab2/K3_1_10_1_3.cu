/**
    Ahmed_Hany_Farouk_1_10

    Sec:    1
    BN:     10
    Code:   9202213

    Ahmed_Sayed_Sayed_1_3

    Sec:    1
    BN:     3
    Code:   9202111
*/

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Perform matrix addition
 * @param C : Resultant matrix (output)
 * @param A : First input matrix
 * @param B : Second input matrix
 * @param n : Number of rows
 * @param m : Number of columns
 */
 __global__ void k3(float *C, float *A, float *B, int n, int m) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < m) {
        for (int i = 0; i < n; ++i) {
            C[i * m + col] = A[i * m + col] + B[i * m + col];
        }
    }
}

int main(int argc, char *argv[]) {

    // Check if the correct number of command-line arguments is provided
    if (argc != 3) {
        fprintf(stderr, "You should use: %s input_filename output_filename\n", argv[0]);
        return 1;
    }

    // Open the input file and check for errors
    FILE *input_file = fopen(argv[1], "r");
    if (!input_file) {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }

    // Open the output file and check for errors
    FILE *output_file = fopen(argv[2], "w");
    if (!output_file) {
        fprintf(stderr, "Error: Cannot open output file %s\n", argv[2]);
        fclose(input_file);
        return 1;
    }

    int t, n, m;
    float *matrix1, *matrix2, *out;
    float *d_matrix1, *d_matrix2, *d_out;
    
    // Read the number of test cases from the input file
    fscanf(input_file, "%d", &t);
    for (int tt = 0; tt < t; tt++) {

        // Read the number of rows and columns for the current test case
        fscanf(input_file, "%d%d", &n, &m);

        // Allocate memory for matrices on the host
        matrix1 = (float*)malloc(sizeof(float) * n * m);
        matrix2 = (float*)malloc(sizeof(float) * n * m);
        out = (float*)malloc(sizeof(float) * n * m);

        // Read the values of matrix1 from the input file
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fscanf(input_file, "%f", &matrix1[i * m + j]);
            }
        }

        // Read the values of matrix2 from the input file
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fscanf(input_file, "%f", &matrix2[i * m + j]);
            }
        }

        // Allocate memory for matrices on the device
        cudaMalloc((void**)&d_matrix1, n * m * sizeof(float));
        cudaMalloc((void**)&d_matrix2, n * m * sizeof(float));
        cudaMalloc((void**)&d_out, n * m * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(d_matrix1, matrix1, n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix2, matrix2, n * m * sizeof(float), cudaMemcpyHostToDevice);

        // Define the number of threads per block
        int threadsPerBlock = 256;

        // Define the number of blocks
        int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        k3<<<numBlocks, threadsPerBlock>>>(d_out, d_matrix1, d_matrix2, n, m);
        
        // Copy result back to host
        cudaMemcpy(out, d_out, sizeof(float) * n * m, cudaMemcpyDeviceToHost);
        
        // Write the result to the output file
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(output_file, "%f ", out[i * m + j]);
            }
            fprintf(output_file, "\n");
        }

        // Free device memory
        cudaFree(d_matrix1);
        cudaFree(d_matrix2);
        cudaFree(d_out);

        // Free host memory
        free(matrix1);
        free(matrix2);
        free(out);
    }

    // Close input and output files
    fclose(input_file);
    fclose(output_file);

    return 0;
}
