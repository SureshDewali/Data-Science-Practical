import numpy as np

# 1️⃣ Broadcasting between different-shaped arrays
print("Name:Suresh Dewali")
print("Roll no:1323575")
a = np.array([[1], [2], [3]])     # shape (3,1)
b = np.array([10, 20, 30])        # shape (3,)
result = a + b
print("Broadcasting Result:\n", result)

# 2️⃣ Generate random arrays and apply statistical computations
random_array = np.random.rand(3, 3)
print("Random Array:\n", random_array)
print("Mean:", np.mean(random_array))
print("Max:", np.max(random_array))
print("Min:", np.min(random_array))

# 3️⃣ Linear algebra operations
matrix = np.array([[2, 3], [1, 4]])
det = np.linalg.det(matrix)
inv = np.linalg.inv(matrix)
eigvals, eigvecs = np.linalg.eig(matrix)

print("Determinant:", det)
print("Inverse:\n", inv)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
