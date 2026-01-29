import numpy as np

# Array and scaler broadcasting
# arr = np.array([1, 2, 3])
# print(arr + 10)  # adds 10 to each and every element.

# matrix = np.array([[1, 2, 3], [4, 5, 6]])
# vector = np.array([[1, 2, 3]])
# print(
#     vector + matrix
# )  # adds vector array to each row of the matrix as it is a 2D matrix.


# Aggregation functions
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print("sum: ", np.sum(arr))
# print("mean: ", np.mean(arr))
# print("maximum : ", np.max(arr))
# print("standard: ", np.std(arr))
# print("sum along rows: ", np.sum(arr, axis=1)) # return sum of each row


# Boolean indexing and filtering
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0, 9])

# evens = arr[arr % 2 == 0]
# print(evens)

# arr[arr > 3] = 0
# print("New array: ", arr)


# Random number generation and random seeds

# random_array = np.random.rand(3, 3)
# print("Random Array: \n", random_array)

# random_integers = np.random.randint(0, 10, size=(2, 6))
# print("Random integer array: \n", random_integers)

# seeding
# np.random.seed(42)  # it will give a fix random numbers
# random_array = np.random.rand(3, 3)
# print("Random Array: \n", random_array)

# random_integers = np.random.randint(0, 10, size=(2, 6))
# print("Random integer array: \n", random_integers)


# Exercise
# Generate random dataset and filter it
# dataset = np.random.randint(1, 51, size=(5, 5))
# print("originbak dataset: \n", dataset)

# filter valuers greater than 25 and replace it with 0
# dataset[dataset > 25] = 0
# print("modified dataset: \n", dataset)


# do aggregation on the dataset
# print("sum: \n", np.sum(dataset))
# print("Mean: \n", np.mean(dataset))
