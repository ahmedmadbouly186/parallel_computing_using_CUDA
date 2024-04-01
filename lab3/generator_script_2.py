import random

def generate_sorted_floats(n, filename):
    with open(filename, 'w') as file:
        for _ in range(n):
            num = round(random.uniform(0, 100), 2)  # Generating random float between 0 and 100
            file.write(f"{num}\n")

def binary_search(value, filename):
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file]
    
    left, right = 0, len(numbers) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if numbers[mid] == value:
            return mid
        elif numbers[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1

if __name__ == "__main__":
    n = int(input("Enter the number of float numbers to generate: "))
    filename = "float_numbers.txt"
    generate_sorted_floats(n, filename)
    
    search_value = float(input("Enter the value to search: "))
    pos = binary_search(search_value, filename)
    
    if pos != -1:
        print(f"Value {search_value} found at position {pos}.")
    else:
        print(f"Value {search_value} not found.")
