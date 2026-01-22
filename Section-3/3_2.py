import numpy as np

# determinant of matrix
# A = np.array([[1, 2], [3, 4]])
# determinant = np.linalg.det(A)
# print("Determinant od A: ", determinant)

# Inverse of matrix
# inverse = np.linalg.inv(A)
# print("Inverse of matrix: \n", inverse)


# Eigen Values and Eigen Vectors

# eigenvalues, eigenvectors = np.linalg.eig(A)
# print("Eigen Values: \n", eigenvalues)
# print("Eigen Vectors: \n", eigenvectors)


# Matrix decomposition
# decomposing the vector in to three parts

# U, S, Vt = np.linalg.svd(A)
# print("U: \n", U)
# print("Singular Values: \n", S)
# print("V Transpose: \n", Vt)


###############EXERCISE#############################
# A = np.array([[2, 3, 5], [6, 9, 8], [2, 6, 4]])
# determinant = np.linalg.det(A)
# inverse = np.linalg.inv(A)

# print("Determinant: \n", determinant)
# print("Inverse: \n", inverse)


# A = np.array([[2, 6], [9, 8]])
# eigenval, eigenVect = np.linalg.eig(A)

# print("Eigen Values: ", eigenval)
# print("Eigen Vectors: ", eigenVect)


A = np.array([[3, 5, 6], [8, 1, 6], [5, 0, 1]])
U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("S: \n", S)
print("Vt: \n", Vt)

sigma = np.zeros((3, 3))
np.fill_diagonal(sigma, S)
reconstructed = U @ sigma @ Vt
print("Reconstructed Matrtix:  \n", reconstructed)
