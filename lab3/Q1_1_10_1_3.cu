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

#define GRID_SIZE 1
#define BLOCK_SIZE 1024

/**
 * @brief get the sum of vector elements
 * @param RSum : Resultant Float (output)
 * @param vec : Input Vector
 * @param n : Size of the vector
 */
__global__ void vectorSum(float *RSum, float *vec, int n)
{
    __shared__ float vecShared[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    while (i < n)
    {
        sum += vec[i];
        i += blockDim.x;
    }
    vecShared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            vecShared[threadIdx.x] += vecShared[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        RSum[0] = vecShared[0];
    }
}

int main(int argc, char *argv[])
{

    // check that there is input and output file
    if (argc != 2)
    {
        fprintf(stderr, "You should use: %s input_filename\n", argv[0]);
        return 1;
    }

    // read the input file and check if can open it
    FILE *input_file = fopen(argv[1], "r");
    if (!input_file)
    {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }

    // Initialize size of the vector to 0
    int n = 0;
    // Count the number of elements
    float number;
    while (fscanf(input_file, "%f", &number) == 1)
    {
        n++;
    }

    // Rewind the file back to the beginning
    rewind(input_file);

    float *vec, *out;
    float *d_vec, *d_out;

    // Allocate memory for the host
    vec = (float *)malloc(sizeof(float) * n);
    out = (float *)malloc(sizeof(float));

    // Read the vector from the file
    for (int i = 0; i < n; i++)
    {
        fscanf(input_file, "%f", &vec[i]);
    }

    /* CUDA */

    // Allocate memory on the device
    cudaMalloc((void **)&d_vec, n * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_vec, vec, n * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke the kernel
    vectorSum<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_vec, n);

    // Copy the result back to the host
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_out);

    // Print the result
    printf("%f", *out);

    // Free host memory
    free(vec);
    free(out);

    // Close The input file
    fclose(input_file);

    return 0;
}
