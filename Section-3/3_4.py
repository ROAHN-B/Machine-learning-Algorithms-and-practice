import sympy as sp

x = sp.symbols("x")
f = x**2
definite_integrat = sp.integrate(f, (x, 0, 2))
indefinite_integrat = sp.integrate(f, x)

print("Indefinite integral: \n", indefinite_integrat)
print("Definite integral: \n", definite_integrat)

# Local minimum and global minimum
# stochastic gradient descent (SGD) and its variants

# convex functions ensures that anny local minimum is also global minimum
