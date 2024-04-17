import time
import numpy as np

# Function to perform matrix multiplication
def matrix_multiply(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

# Create two random matrices
matrix_size = 100
matrix1 = np.random.rand(matrix_size, matrix_size)
matrix2 = np.random.rand(matrix_size, matrix_size)

# Perform matrix multiplication without parallelism
start_time = time.time()
result = matrix_multiply(matrix1, matrix2)
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print(f"Matrix multiplication without parallelism took {execution_time} seconds.")
