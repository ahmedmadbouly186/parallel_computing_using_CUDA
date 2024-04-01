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
 * @brief Binary search for an eleemnt in the vector
 * @param pos : position (output) -1 if not found, otherwise the index of element in the array
 * @param val : value to be searched
 * @param vec : Input Vector
 * @param n : Size of the vector
 */
__global__ void vectorSearch(int *pos, float *val, float *vec, int n)
{
    __shared__ int index;
    if (threadIdx.x == 0)
        index = -1;

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int segment_size = (n + gridDim.x - 1) / gridDim.x;
    int start = tid * segment_size;
    int end = ((tid + 1) * segment_size < n) ? (tid + 1) * segment_size : n;

    // Perform binary search on the segment
    int left = start;
    int right = end - 1;

    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (vec[mid] == val[0])
        {
            index = mid; // Update index if element found
            break;
        }
        else if (vec[mid] < val[0])
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        pos[0] = index; // Update pos with the found index
    }
}



int main(int argc, char *argv[])
{

    // check that there is input and output file
    if (argc != 3)
    {
        fprintf(stderr, "You should use: %s input_filename vlaue\n", argv[0]);
        return 1;
    }

    // read the input file and check if can open it
    FILE *input_file = fopen(argv[1], "r");
    if (!input_file)
    {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }

    // get the value to search for
    float value = atof(argv[2]);

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

    float *vec;
    int *out;
    float *d_vec, *d_value;
    int *d_out;

    // Allocate memory for the host
    vec = (float *)malloc(sizeof(float) * n);
    out = (int *)malloc(sizeof(int));

    // Read the vector from the file
    for (int i = 0; i < n; i++)
    {
        fscanf(input_file, "%f", &vec[i]);
    }

    /* CUDA */

    // Allocate memory on the device
    cudaMalloc((void **)&d_vec, n * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(int));
    cudaMalloc((void **)&d_value, sizeof(float));

    // Copy the data to the device
    cudaMemcpy(d_vec, vec, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice);

    // Invoke the kernel
    vectorSearch<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_value, d_vec, n);

    // Copy the result back to the host
    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vec);
    cudaFree(d_out);
    cudaFree(d_value);

    // Print the result
    printf("%d", *out);

    // Free host memory
    free(vec);
    free(out);

    // Close The input file
    fclose(input_file);

    return 0;
}
