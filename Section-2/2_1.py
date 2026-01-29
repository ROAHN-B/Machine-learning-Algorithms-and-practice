# # numpy arrays are effcient in complex mathematical calculations
import numpy as np

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(arr)

# zeroes = np.zeros([3, 3])
# print(zeroes)

# ones = np.ones([2, 4])
# print(ones)

# range_array = np.arange(1, 10, 2)
# print(range_array)

# line_space = np.linspace(1, 5, 8)
# print(line_space)


# arr = np.array([1, 2, 3, 4, 5, 6])
# reshaped = arr.reshape((2, 3))
# print(reshaped)  # converting one dimentional array into 2-D matrix

# arr1 = np.array([1, 2, 3])
# expanded = arr1[:, np.newaxis]
# print(expanded)

# ------------BASIC OPERATIONS ON ARRAYS------------------------

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a + b)
# print(a * b)
# print(a - b)
# print(a / b)

# arr = np.array([4, 16, 25])
# print(np.max(arr))


# ------------------INDEXING, RESHASPING, SLICING----------------------

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(a[-1])
# print(a[5:2:-1])
# print(a.reshape(3, 3))  # convert to multidimentional matrix

# -----------------EXERCISES---------------------------


# EXERCISE -1
# a = np.arange(1, 6)
# b = np.arange(6, 11)


# print("1st array: ", a)
# print("2nd arary: ", b)
# print("addition: ", a + b)
# print("subtraction: ", a - b)
# print("multiplication: ", a * b)
# print("division: ", a / b)


# EXERCISE -2
# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print("original matrix: \n", matrix)
# print("transpose matrix: \n", np.transpose(matrix))


# EXERCISE -3
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
print("Addition: \n", matrix1 + matrix2)
print("subtraction: \n", matrix1 - matrix2)
print("multiplication: \n", matrix1 * matrix2)
print("division: \n", matrix1 / matrix2)
