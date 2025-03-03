import numpy as np
import matplotlib.pyplot as plt

# Data: Scores from discrete mathematics and general math tests
discrete = np.array([12, 26, 38, 40, 60, 69, 15, 25, 26, 17])
math = np.array([78, 68, 51, 60, 53, 42, 66, 30, 68, 33])

# Mean Squared Error function
def msqe(y, predictedy):
    return np.mean((y - predictedy) ** 2)

# Linear search for best 'm' in y = m * x
searching_space = np.linspace(-3, 3, 50)
err = [msqe(math, m * discrete) for m in searching_space]

# Find 'm' that minimizes the error
lowestm = searching_space[np.argmin(err)]
print(f"Best m (Linear Search): {lowestm:.4f}")
print(f"Minimum MSE: {min(err):.4f}")

# Gradient Descent for Linear Model
def gradient_descent_linear(x, y, lr=0.0001, iterations=50):
    m = 0.1  # Initial guess
    n = len(x)
    
    for _ in range(iterations):
        predictedy = m * x
        grad = (-2 / n) * np.sum(x * (y - predictedy))  # Compute gradient
        m -= lr * grad  # Update step
        if abs(grad) < 1e-6:
            break
    
    return m

# Apply gradient descent
best_m_gd = gradient_descent_linear(discrete, math)
print(f"Best m (Gradient Descent): {best_m_gd:.4f}")

# Polynomial Function Optimization using Gradient Descent
np.random.seed(42)
X = np.linspace(-10, 10, 400)
y_poly = 4*X**2 - 10*X + np.random.normal(0, 50, X.shape)

# Derivative: dy/dx of 4X² - 10X
def gradient(x):
    return 8 * x - 10  # Corrected to match function

# Gradient Descent to find local minima
def gradient_descent(start_x, learning_rate=0.01, iterations=1000):
    x = start_x
    for _ in range(iterations):
        grad = gradient(x)
        x -= learning_rate * grad
        if abs(grad) < 1e-5:
            break
    return x

# Try different starting points
initial_points = [-1, 0, 1]
minima = [gradient_descent(x0) for x0 in initial_points]

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot error values vs. 'm'
axes[0].plot(searching_space, err, label="MSE vs m", color='blue')
axes[0].axvline(lowestm, color='r', linestyle="--", label=f"Best m = {lowestm:.4f}")
axes[0].set_xlabel("m values")
axes[0].set_ylabel("MSE")
axes[0].set_title("Finding Best m value")
axes[0].legend()
axes[0].grid()

# Polynomial function plot with found minima
axes[1].plot(X, 4*X**2 - 10*X, label='Function')
axes[1].scatter(minima, [4*x**2 - 10*x for x in minima], color='red', label='Local Minima', zorder=3)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('Finding Local Minima using Gradient Descent')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
