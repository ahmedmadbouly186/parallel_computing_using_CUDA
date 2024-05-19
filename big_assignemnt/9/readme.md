# step1 : install cuda

- !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
- %load_ext nvcc4jupyter

# step2: run test generator :

- python .\generator.py num_nodes

# step3: compile and run cpu solver

- gcc tsp_cpu_solver.c -o tsp_solver -lm
- ./tsp_solver edges.txt

# step4: copile and run the cuda:

- nvcc tsp_gpu_solver.cu -o run -rdc=true -lcudadevrt
- nvprof ./run edges.txt
