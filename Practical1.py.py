# Import NumPy library
import numpy as np

# 1️⃣ Create 1D and 2D arrays
print("Name:Suresh Dewali")
print("Roll no:1323575")
arr1 = np.array([10, 20, 30, 40, 50])          # 1D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])        # 2D array

print("1D Array:\n", arr1)
print("2D Array:\n", arr2)

# 2️⃣ Indexing and Slicing
print("Element at index 2 in arr1:", arr1[2])
print("Sliced elements from arr1 (1 to 4):", arr1[1:4])
print("Element from 2D array arr2[1,2]:", arr2[1, 2])

# 3️⃣ Element-wise operations
print("Addition:", arr1 + 5)
print("Multiplication:", arr1 * 2)

# 4️⃣ Matrix operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix Addition:\n", matrix1 + matrix2)
print("Matrix Multiplication:\n", np.dot(matrix1, matrix2))

# 5️⃣ Using NumPy functions
print("Mean of arr1:", np.mean(arr1))
print("Standard Deviation of arr1:", np.std(arr1))
print("Dot Product of arr1 with itself:", np.dot(arr1, arr1))
