#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda.h>

#define ull unsigned long long

__device__ void ithpermutation(ull n,ull i,ull* fact,ull*path){
        long long j, k = 0;
        for (k = 0; k < n; ++k) {
            path[k] = i / fact[n - 1 - k];
            i = i % fact[n - 1 - k];
        }

        // Adjust values to obtain the permutation
        for (k = n - 1; k > 0; --k)
            for (j = k - 1; j >= 0; --j)
                if (path[j] <= path[k])
                    path[k]++;

}
__device__ int lock_me(int* mutex, int id) {
if (atomicCAS((int*) (mutex + id), 0, 1) == 0)
  return 1;
return 0;
}


__device__ void unlock_me(int* mutex, int id) {
    atomicExch((int*) (mutex + id), 0);
}

__global__ void calculatePathWeight(double *graph, ull *fact, double *min_path,int* best_path,int *lock, ull numPerms, ull numVertices) {
    ull idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPerms) return;

    // prepare shared memory
    extern __shared__ ull shared_array[];
    ull* shared_factorial=(ull*)&shared_array[0];
    double *shared_graph=(double*)&shared_array[numVertices+1];
    
    int fact_load_thread = (numVertices+1 + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < fact_load_thread; i++) {
        if(threadIdx.x*fact_load_thread+i<=numVertices){
            shared_factorial[threadIdx.x*fact_load_thread+i]=fact[threadIdx.x*fact_load_thread+i];
        }
    }
    int graph_elements = numVertices*numVertices;
    int graph_load_thread = (graph_elements + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < graph_load_thread; i++) {
        if(threadIdx.x*graph_load_thread+i<=graph_elements){
            shared_graph[threadIdx.x*graph_load_thread+i]=graph[threadIdx.x*graph_load_thread+i];
        }
    }
    // if(threadIdx.x <=  numVertices){
    //     shared_factorial[threadIdx.x]=fact[threadIdx.x];
    // }
    // if(threadIdx.x==0){
    //     // for (ull i = 0; i <= numVertices; ++i) {
    //     //     shared_factorial[i]=fact[i];
    //     // }
    //    for (int i = 0; i < numVertices; i++) {
    //         for (int j = 0; j < numVertices; j++) {
    //             shared_graph[i * numVertices + j] = graph[i * numVertices + j];
    //         }
    //     }
    // }
    
    __syncthreads();

    ull total_threads = blockDim.x * gridDim.x;
    ull permutatins_per_thread = (numPerms + total_threads -1) / total_threads;
    ull *path = new ull[numVertices];
    
    ull *best_path_thread = new ull[numVertices];
    double best_cost_thread = min_path[0];
    for(ull path_num=idx*permutatins_per_thread ; path_num < (idx+1)*permutatins_per_thread ; path_num++)
    {
        if(path_num>=numPerms)break;

        ithpermutation(numVertices,path_num,shared_factorial,path);
        double current_pathweight = 0;
        for (ull j = 1; j < numVertices; j++) {
            ull u = path[j - 1];
            ull v = path[j];
            current_pathweight += shared_graph[u * numVertices + v];
        }
        ull s = path[0];
        ull k = path[numVertices - 1];
        current_pathweight += shared_graph[k * numVertices + s];
        
        if(current_pathweight < best_cost_thread){
            best_cost_thread = current_pathweight;
            for (int i = 0; i < numVertices; i++) {
                best_path_thread[i] = path[i];
            }
         }


    }
    
    // while (atomicExch(lock, 1) != 0);
    // while (atomicCAS(lock, 0, 1) != 0);
    // lock =1;
    // lock_me(lock,0);
    int successfull = 0;
   while (!successfull){
    if (lock_me(lock, 0)) { //lock acquired?
        unlock_me(lock, 0); // then unlock
        successfull = 1;}
    }



    if(best_cost_thread<min_path[0]){
        min_path[0] = best_cost_thread;
        for (int i = 0; i < numVertices; i++) {
            best_path[i] = best_path_thread[i];
        }
    }

    unlock_me(lock,0);
    // lock=0;
    // atomicExch(lock, 0);
    //atomicMin_double(min_path, current_pathweight);

    delete[] path;
}
// Function to read the graph from a file
__host__ void readGraphFromFile(FILE *file, double** graph) {
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }

    int u, v;
    double w;
    while (fscanf(file, "%d %d %lf", &u, &v, &w) == 3) {
        graph[u][v] = w;
        graph[v][u] = w;
    }

    fclose(file);
}
// Function to calculate the factorial array
__host__ void calculateFactorials(ull n, ull* fact) {
    fact[0] = 1;
    for (ull i = 1; i <= n; ++i) {
        fact[i] = fact[i - 1] * i;
    }
}

int main(int argc, char **argv) {
    
    char*file_name;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }
    file_name = argv[1];
    // file_name="edges.txt";
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }
    int N;
    fscanf(file,"%d",&N);
    double** graph = (double**)malloc(N * sizeof(double*));
    for(int i=0;i<N;i++){
        graph[i]=(double*)malloc(N * sizeof(double));
    }
    // Initialize the graph with 0s for diagonal and infinity for other entries
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                graph[i][j] = 0;
            } else {
                graph[i][j] = DBL_MAX;
            }
        }
    }
    // Read the graph from the file
    readGraphFromFile(file, graph);

    double *h_graph = (double *)malloc(N * N * sizeof(double));
    double *d_graph;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_graph[i * N + j] = graph[i][j];
        }
    }
    cudaMalloc(&d_graph, N * N * sizeof(double));
    cudaMemcpy(d_graph, h_graph, N * N * sizeof(double), cudaMemcpyHostToDevice);

    ull *h_fact = (ull *)malloc((N + 1) * sizeof(ull));
    ull *d_fact;
    calculateFactorials((ull)N, h_fact);
    cudaMalloc(&d_fact, (N + 1) * sizeof(ull));
    cudaMemcpy(d_fact, h_fact, (N + 1) * sizeof(ull), cudaMemcpyHostToDevice);

    double *d_min_path;
    cudaMalloc(&d_min_path, sizeof(double));
    double h_min_path = DBL_MAX;
    cudaMemcpy(d_min_path, &h_min_path, sizeof(double), cudaMemcpyHostToDevice);

    int *h_best_path = (int *)malloc(N * sizeof(int));
    int *d_best_path;
    cudaMalloc(&d_best_path, N * sizeof(int));
    
    int h_lock = 0;
    int *d_lock;
    cudaMalloc(&d_lock, sizeof(int));
    cudaMemcpy(d_lock, &h_lock, sizeof(int), cudaMemcpyHostToDevice);

    ull numPerms = h_fact[N];
    ull blockSize = 1024;
    // ull numBlocks = (numPerms + blockSize - 1) / blockSize;


    calculatePathWeight<<<10, blockSize, N * N * sizeof(double)   +  (N + 1) * sizeof(ull)   >>>(d_graph, d_fact, d_min_path,d_best_path,d_lock, numPerms,(ull) N);

    cudaMemcpy(h_best_path, d_best_path, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min_path, d_min_path, sizeof(double), cudaMemcpyDeviceToHost);
    printf("\n\nMinimum Path Weight: %lf\n\n", h_min_path);
    printf("Best Path: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_best_path[i]);
    }
    printf("\n");
    cudaFree(d_best_path);
    cudaFree(d_graph);
    cudaFree(d_fact);
    cudaFree(d_min_path);
    free(h_graph);
    free(h_fact);
    free(h_best_path);

    return 0;
}
