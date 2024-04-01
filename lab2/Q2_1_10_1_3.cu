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
 * @brief Perform matrix–vector multiplication
 * @param d_out     : Resultant vector (output)
 * @param d_matrix  : input matrix
 * @param d_vec     : input vector
 * @param n         : Number of rows
 * @param m         : Number of columns
 */
__global__ void matrix_mul(float *d_out, float *d_matrix, float *d_vec, int n, int m) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float temp_out = 0;
    for (int j = 0; j < m; ++j) {
      temp_out += d_matrix[j+m*row] * d_vec[j];
    }
    d_out[row] = temp_out;
  }
}

int main(int argc, char *argv[]) {

  // Check if the correct number of command-line arguments is provided
  if (argc != 3) {
    fprintf(stderr, "You should use: %s input_filename output_filename\n", argv[0]);
    return 1;
  }

  freopen(argv[1], "r", stdin);
  freopen(argv[2], "w", stdout);

  int t, n, m;
  float *matrix;
  float *vec,*out;
  float *d_matrix;
  float *d_vec,*d_out;

  // Read the number of test cases from the input file
  scanf("%d",&t);
  for(int tt=0;tt<t;tt++){

  // Read the number of rows and columns for the current test case
  scanf("%d%d",&n,&m);

  // Allocate memory for matrix and vector on the host
  vec = (float*)malloc(sizeof(float) * m);
  matrix = (float*)malloc(sizeof(float*) * n *m);
  out = (float*)malloc(sizeof(float) * n);

  // Read the values of matrix from the input file
  for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
      scanf("%f",&matrix[i * m+ j]);
    }
  }

  // Read the values of vector from the input file
  for(int j=0;j<m;j++){
      scanf("%f",&vec[j]);
  }

  // Copy data from host to device
  cudaMalloc((void**)&d_matrix, n * m * sizeof(float*));  // Allocate memory for pointers to rows on the device

  // // Allocate memory for the vector and output
  cudaMalloc((void**)&d_vec, m * sizeof(float));
  cudaMalloc((void**)&d_out, n * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_matrix, matrix, m *n * sizeof(float), cudaMemcpyHostToDevice);  // Copy vec
  cudaMemcpy(d_vec, vec, m * sizeof(float), cudaMemcpyHostToDevice);  // Copy vec

  // Define the number of threads per block and the number of blocks
  int threadsPerBlock = 256;
  int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  matrix_mul<<<numBlocks,threadsPerBlock>>>(d_out, d_matrix, d_vec, n,m);
  
  cudaMemcpy(out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);
  for(int i=0;i < n;i++){
    printf("%f\n",out[i]);
  }

  
  cudaFree(d_matrix);  
  cudaFree(d_vec);
  cudaFree(d_out);

  // Deallocate memory on the host
  free(vec);
  free(matrix); 
  free(out);

  }
}