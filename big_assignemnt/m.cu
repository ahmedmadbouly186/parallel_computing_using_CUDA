#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

_device_ int d_binarySearch(double* arr, double target, int left, int right) {
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target)
      return mid;
    if (arr[mid] < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

_global_ void parallelSearch(double* arr, double target, int* target_index, int arr_size, int stride) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Check if within bounds
  if (tid < arr_size) {
    // Initialize local variables
    int start = tid * stride;
    int end = min((tid + 1) * stride - 1, arr_size - 1);
    int local_index = -1;

    // Perform binary search
    if (arr[start] <= target && arr[end] >= target)
      local_index = d_binarySearch(arr, target, start, end);

    // Store result in global memory
    if (local_index != -1 && atomicCAS(target_index, -1, local_index) == -1)
      *target_index = local_index;
  }
}

void binarySearch(double* h_array, double target, int arr_size, int* h_target_index, int stride = 8) {
  double* d_array;
  int* d_target_index;

  cudaMalloc(&d_array, arr_size * sizeof(double));
  cudaMemcpy(d_array, h_array, arr_size * sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc(&d_target_index, sizeof(int));
  int notFound = -1;
  cudaMemcpy(d_target_index, &notFound, sizeof(int), cudaMemcpyHostToDevice);

  // Assuming 1 block and 1024 threads per block as per requirement
  int threadsPerBlock = 1024;
  int blocksPerGrid = 1;

  // printf("Launching Parallel Binary Search kernel with %d blocks and %d threads per block with %d stride!\n", blocksPerGrid, threadsPerBlock, stride);

  // Launch the kernel
  parallelSearch<<<blocksPerGrid, threadsPerBlock>>>(d_array, target, d_target_index, arr_size, stride);
  cudaDeviceSynchronize();

  // Copy the result back to the host
  cudaMemcpy(h_target_index, d_target_index, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_array);
  cudaFree(d_target_index);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_file> <target>\n", argv[0]);
    return 1;
  }

  double target = atof(argv[2]);

  FILE *file;
  file = fopen(argv[1], "r");

  if (!file) {
    printf("Error opening file!\n");
    return 1;
  }

  int num_elements = 0;
  double temp;
  while (fscanf(file, "%lf", &temp) == 1) {
    num_elements++;
  }

  fseek(file, 0, SEEK_SET);

  double h_array = (double)malloc(num_elements * sizeof(double));
  if (!h_array) {
    printf("Memory allocation failed!\n");
    fclose(file);
    return 1;
  }

  for (int i = 0; i < num_elements; ++i) {
    fscanf(file, "%lf", h_array + i);
  }

  fclose(file);

  printf("h_array[173638] = %lf\n", h_array[173638]);

  int result = -1;
  binarySearch(h_array, target, num_elements, &result, 8);

  if (result != -1) {
    printf("%d\n", result);
  } else {
    printf("-1\n");
  }

  free(h_array);

  return 0;
}