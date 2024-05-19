# Python3 program to implement traveling salesman 
# problem using naive approach. 
import sys
from sys import maxsize 
import random
from itertools import permutations
V = 4
debug = False
excute = False
# implementation of traveling Salesman Problem 
def travellingSalesmanProblem(graph, s): 

	# store all vertex apart from source vertex 
	vertex = [] 
	for i in range(V): 
		if i != s: 
			vertex.append(i) 

	# store minimum weight Hamiltonian Cycle 
	min_path = maxsize 
	next_permutation=permutations(vertex)
	# print all permutations

	for i in next_permutation:
		# print(i)
		# store current Path weight(cost) 
		current_pathweight = 0
        
        
		# compute current path weight 
		k = s 
		
		for j in i: 
			current_pathweight += graph[k][j] 
			k = j 
		current_pathweight += graph[k][s] 

		# update minimum 
		min_path = min(min_path, current_pathweight) 
		
	return min_path 
def test_generator(n, filename):
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for u in range(n):
            for v in range(u + 1, n):
                w = random.uniform(1, 100)
                f.write(f"{u} {v} {w:.2f}\n")

def read_edges_from_file(filename):
    edges = []
    with open(filename, 'r') as f:
        # Skip the first line
        next(f)
        for line in f:
            u, v, w = line.split()
            edges.append((int(u), int(v), float(w)))
    return edges

def create_graph(n, edges):
    graph = [[0 if i == j else sys.maxsize for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        graph[u][v] = w
        graph[v][u] = w
    return graph

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <number_of_vertices>")
        sys.exit(1)
    
    V = int(sys.argv[1])
    filename = "edges.txt"

    # Generate a random test case and save to file
    test_generator(V, filename)
    if excute :
      # Read the edges from the file
      edges = read_edges_from_file(filename)
      if debug:
          print("Read Edges from file:")
          for u, v, w in edges:
              print(f"{u} {v} {w:.2f}")

      # Create a graph from the edges
      graph = create_graph(V, edges)
      if debug:
          print("Graph:")
          for vec in graph:
              print(vec)
    # set the sol to maxmimum value of int 
      sol = sys.maxsize
    # print(travellingSalesmanProblem(graph, s))
      for s in range(V):
        sol=min(sol,travellingSalesmanProblem(graph, s))
      print("the best solution is : ",sol)