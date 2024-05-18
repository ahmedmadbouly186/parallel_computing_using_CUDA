#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define debug 0
#define ull unsigned long long

void kthPermutation(const ull n, ull i, ull *fact, ull *perm) {

    long long j, k = 0;
    for (k = 0; k < n; ++k) {
        perm[k] = i / fact[n - 1 - k];
        i = i % fact[n - 1 - k];
    }

    // Adjust values to obtain the permutation
    for (k = n - 1; k > 0; --k)
        for (j = k - 1; j >= 0; --j)
            if (perm[j] <= perm[k])
                perm[k]++;

}

// Function to calculate the minimum Hamiltonian cycle
double travellingSalesmanProblem(double **graph, ull N,int *best_path) {
    // Store the minimum weight Hamiltonian Cycle
    double min_path = DBL_MAX;

    // Generate all permutations of vertices
    ull perm_count = 1;
    ull *fact = (ull *)calloc(N + 1, sizeof(ull));
    ull *path = (ull *)calloc(N, sizeof(ull));
    fact[0] = 1;

    for (ull i = 1; i <= N; i++) {
        perm_count *= i;
        fact[i] = perm_count;
    }

    // Calculate the minimum path weight
    for (ull i = 0; i < perm_count; i++) {
        kthPermutation(N, i, fact, path);

        double current_pathweight = 0;

        // Compute current path weight
        for (ull j = 1; j < N; j++) {
            ull u = path[j - 1];
            ull v = path[j];
            current_pathweight += graph[u][v];
        }
        ull s = path[0];
        ull k = path[N - 1];
        current_pathweight += graph[k][s];

        // Update minimum path weight
        if (current_pathweight < min_path) {
            min_path = current_pathweight;
            for (ull j = 0; j < N; j++) {
                best_path[j] = path[j];
            }
        }
    }

    free(fact);
    free(path);

    return min_path;
}

// Function to read the graph from a file
void readGraphFromFile(FILE *file, double **graph) {
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

// Driver function
int main(int argc, char *argv[]) {
    printf("TSP CPU Solver\n");
    char *file_name;
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }
    file_name = argv[1];
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return EXIT_FAILURE;
    }

    int N;
    fscanf(file, "%d", &N);
    double **graph = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        graph[i] = (double *)malloc(N * sizeof(double));
        if (graph[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return EXIT_FAILURE;
        }
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

    if (debug) {
        printf("Graph:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f  ", graph[i][j]);
            }
            printf("\n");
        }
    }
    // Find the minimum path using TSP
    int * best_path = (int *)malloc(N * sizeof(int));
    double sol = travellingSalesmanProblem(graph, N,best_path);
    printf("The best solution is : %lf\n", sol);
    printf("The best path is : ");
    for (int i = 0; i < N; i++) {
        printf("%d ", best_path[i]);
    }
    printf("\n");
    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(graph[i]);
    }
    free(graph);

    return EXIT_SUCCESS;
}
