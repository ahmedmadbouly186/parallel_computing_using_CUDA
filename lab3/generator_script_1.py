import random


def generate_random_floats(n, min_val=0.0, max_val=100.0):
    """Generate n random float numbers between min_val and max_val."""
    return [random.uniform(min_val, max_val) for _ in range(n)]


def write_floats_to_file(numbers, filename):
    """Write list of floats to a file."""
    with open(filename, 'w') as file:
        for number in numbers:
            file.write(str(number) + '\n')


def calculate_sum(numbers):
    """Calculate the sum of numbers."""
    return sum(numbers)


def write_sum_to_file(sum_val, filename):
    """Write sum to a file."""
    with open(filename, 'w') as file:
        file.write(str(sum_val))


def main():
    # n = int(input("Enter the number of random float numbers to generate: "))
    n = 1000
    random_floats = generate_random_floats(n)

    # Write random floats to a file
    write_floats_to_file(random_floats, 'random_floats.txt')

    # Calculate sum
    sum_of_floats = calculate_sum(random_floats)

    # Write sum to another file
    write_sum_to_file(sum_of_floats, 'sum_of_floats.txt')

    print("Random floats and their sum have been written to files.")


if __name__ == "__main__":
    main()
