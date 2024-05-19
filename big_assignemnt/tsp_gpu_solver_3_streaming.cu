#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda.h>

#define ull unsigned long long

__device__ void ithpermutation(ull n, ull i, ull *fact, ull *path)
{
    long long j, k = 0;
    for (k = 0; k < n; ++k)
    {
        path[k] = i / fact[n - 1 - k];
        i = i % fact[n - 1 - k];
    }

    // Adjust values to obtain the permutation
    for (k = n - 1; k > 0; --k)
        for (j = k - 1; j >= 0; --j)
            if (path[j] <= path[k])
                path[k]++;
}
__device__ int lock_me(int *mutex, int id)
{
    // Atomic Compare and Swap
    if (atomicCAS((int *)(mutex + id), 0, 1) == 0)
        return 1;
    return 0;
}

__device__ void unlock_me(int *mutex, int id)
{
    atomicExch((int *)(mutex + id), 0);
}

// Function to read the graph from a file
__host__ void readGraphFromFile(FILE *file, double *graph, int N)
{
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }

    int u, v;
    double w;
    while (fscanf(file, "%d %d %lf", &u, &v, &w) == 3)
    {
        graph[u * N + v] = w;
        graph[v * N + u] = w;
    }

    fclose(file);
}
// Function to calculate the factorial array
__host__ void calculateFactorials(ull n, ull *fact)
{
    fact[0] = 1;
    for (ull i = 1; i <= n; ++i)
    {
        fact[i] = fact[i - 1] * i;
    }
}
__global__ void calculatePathWeight(double *graph, ull *fact, double *min_path, int *best_path, int *lock, ull start_idx, ull start_perm, ull end_perm, ull numVertices, double *min_paths, short *best_paths)
{
    ull numPerms = end_perm - start_perm;
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;
    min_paths[idx] = DBL_MAX;
    if (idx >= numPerms)
    {

        return;
    }

    // prepare shared memory
    extern __shared__ ull shared_array[];
    ull *shared_factorial = (ull *)&shared_array[0];
    double *shared_graph = (double *)&shared_array[numVertices + 1];

    int fact_load_thread = (numVertices + 1 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < fact_load_thread; i++)
    {
        if (threadIdx.x * fact_load_thread + i <= numVertices)
        {
            shared_factorial[threadIdx.x * fact_load_thread + i] = fact[threadIdx.x * fact_load_thread + i];
        }
    }
    int graph_elements = numVertices * numVertices;
    int graph_load_thread = (graph_elements + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < graph_load_thread; i++)
    {
        if (threadIdx.x * graph_load_thread + i <= graph_elements)
        {
            shared_graph[threadIdx.x * graph_load_thread + i] = graph[threadIdx.x * graph_load_thread + i];
        }
    }
    __syncthreads();

    ull total_threads = blockDim.x * gridDim.x;
    ull permutatins_per_thread = (numPerms + total_threads - 1) / total_threads;
    ull *path = new ull[numVertices];

    short *best_path_thread = &best_paths[(idx + start_idx) * numVertices];
    double best_cost_thread = min_path[0];
    for (ull path_num = idx * permutatins_per_thread; path_num < (idx + 1) * permutatins_per_thread; path_num++)
    {
        if (path_num >= numPerms)
            break;
        ull actual_path_num = path_num + start_perm;

        ithpermutation(numVertices, actual_path_num, shared_factorial, path);
        double current_pathweight = 0;
        for (ull j = 1; j < numVertices; j++)
        {
            ull u = path[j - 1];
            ull v = path[j];
            current_pathweight += shared_graph[u * numVertices + v];
        }
        ull s = path[0];
        ull k = path[numVertices - 1];
        current_pathweight += shared_graph[k * numVertices + s];

        if (current_pathweight < best_cost_thread)
        {
            best_cost_thread = current_pathweight;
            for (int i = 0; i < numVertices; i++)
            {
                best_path_thread[i] = (short)path[i];
            }
        }
    }
    // printf("Thread %d: %lf\n", idx, best_cost_thread);
    min_paths[idx + start_idx] = best_cost_thread;

    delete[] path;
}
__global__ void chooseBestPath(ull total_threads, double *min_path, int *best_path, ull numVertices, double *min_paths, short *best_paths)
{
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        double min = DBL_MAX;
        int min_idx = 0;
        for (int i = 0; i < total_threads; i++)
        {
            if (min_paths[i] < min)
            {
                min = min_paths[i];
                min_idx = i;
            }
        }
        min_path[0] = min;
        short *best_path_thread = &best_paths[min_idx * numVertices];
        for (int i = 0; i < numVertices; i++)
        {
            best_path[i] = best_path_thread[i];
        }
    }
}
int main(int argc, char **argv)
{

    char *file_name;
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }
    file_name = argv[1];
    // file_name="edges.txt";
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }
    int N;
    fscanf(file, "%d", &N);

    // Read the graph from the file

    double *h_graph;
    ull *h_fact;
    int *h_best_path;
    double h_min_path = DBL_MAX;
    int h_lock = 0;

    // allocate the host memory using cudaMallocHost, to allocate it in the pinned memory which is faster to access
    // pinned memory is faster to access than pageable memory because it is directly accessible from the direct memory access (DMA) engine of the GPU
    // The only difference is that the allocated memory cannot be paged by the OS
    cudaMallocHost((void **)&h_fact, (N + 1) * sizeof(ull));
    cudaMallocHost((void **)&h_best_path, N * sizeof(int));
    cudaMallocHost((void **)&h_graph, N * N * sizeof(double));
    cudaDeviceSynchronize();
    readGraphFromFile(file, h_graph, N);
    calculateFactorials((ull)N, h_fact);

    double *d_graph;    // device graph
    ull *d_fact;        // device fact
    double *d_min_path; // device min value
    int *d_best_path;   // device best path
    int *d_lock;        // device lock
    const int nStreams = 3;

    double *d_min_paths; // device min paths
    short *d_best_paths; // device best paths for all threads
    cudaMalloc(&d_best_paths, blockSize * numBlocks * N * nStreams * sizeof(short));
    cudaMalloc(&d_min_paths, blockSize * numBlocks * nStreams * sizeof(double));

    cudaMalloc(&d_best_path, N * sizeof(int));
    cudaMalloc(&d_graph, N * N * sizeof(double));
    cudaMalloc(&d_fact, (N + 1) * sizeof(ull));
    cudaMalloc(&d_min_path, sizeof(double));
    cudaMalloc(&d_lock, sizeof(int));

    ull shred_fac_size = (N + 1) * sizeof(ull);
    ull shred_graph_size = N * N * sizeof(double);
    ull shared_mem_size = shred_fac_size + shred_graph_size;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    // loop over copy, loop over kernel, loop over copy
    for (int i = 0; i < nStreams; ++i)
    {
        cudaMemcpyAsync(d_graph, h_graph, N * N * sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_fact, h_fact, (N + 1) * sizeof(ull), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_min_path, &h_min_path, sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_lock, &h_lock, sizeof(int), cudaMemcpyHostToDevice, stream[i]);
    }
    ull numPerms = h_fact[N];
    ull num_perm_per_stream = (numPerms + nStreams - 1) / nStreams;
    const ull blockSize = 1024;
    const ull numBlocks = 5;
    for (int i = 0; i < nStreams; ++i)
    {
        ull start_perm = i * num_perm_per_stream;
        ull end_perm = (i + 1) * num_perm_per_stream;

        ull start_idx = i * numBlocks * blockSize;
        if (end_perm > numPerms)
            end_perm = numPerms;
        calculatePathWeight<<<numBlocks, blockSize, shared_mem_size, stream[i]>>>(d_graph, d_fact, d_min_path, d_best_path, d_lock, start_idx, start_perm, end_perm, (ull)N, d_min_paths, d_best_paths);
    }
    cudaDeviceSynchronize(); // Synchronize across all blocks
    chooseBestPath<<<1, 1, 0>>>(numBlocks * blockSize * nStreams, d_min_path, d_best_path, N, d_min_paths, d_best_paths);
    cudaDeviceSynchronize(); // Synchronize across all blocks
    for (int i = 0; i < nStreams; ++i)
    {
        cudaMemcpyAsync(h_best_path, d_best_path, N * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(&h_min_path, d_min_path, sizeof(double), cudaMemcpyDeviceToHost, stream[i]);
    }
    // Synchronize the streams to ensure all operations are complete
    for (int i = 0; i < nStreams; ++i)
    {
        cudaStreamSynchronize(stream[i]);
    }

    // Destroy the streams
    for (int i = 0; i < nStreams; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }

    printf("\n\nMinimum Path Weight: %lf\n\n", h_min_path);
    printf("Best Path: ");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_best_path[i]);
    }
    printf("\n");

    cudaFree(d_best_path);
    cudaFree(d_graph);
    cudaFree(d_fact);
    cudaFree(d_min_path);
    cudaFree(d_lock);
    cudaFreeHost(h_graph);
    cudaFreeHost(h_fact);
    cudaFreeHost(h_best_path);

    return 0;
}
