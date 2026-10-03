"""
Project 8: Gradient Descent Implementation
Binary classification with logistic regression, trained from scratch using NumPy.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# TASK 1: Dataset + parameter initialization
# ---------------------------------------------------------------
np.random.seed(42)
n = 100
# Two classes: points around (2, 2) -> class 0 and (4, 4) -> class 1
X0 = np.random.randn(n, 2) * 0.8 + np.array([2, 2])
X1 = np.random.randn(n, 2) * 0.8 + np.array([4, 4])
X = np.vstack([X0, X1])
y = np.array([0] * n + [1] * n)

# Initialize parameters (weights and bias) to zero
w = np.zeros(2)
b = 0.0
print(f"Initial parameters: w = {w}, b = {b}")

# ---------------------------------------------------------------
# TASK 2: Gradient Descent
# ---------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_fn(y, p):
    """Binary cross-entropy loss."""
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def accuracy(y, p):
    return np.mean((p >= 0.5) == y)

learning_rate = 0.1
epochs = 200
m = len(y)

loss_history, acc_history = [], []
w_history, b_history = [], []

for epoch in range(epochs):
    # Forward pass: predictions
    p = sigmoid(X @ w + b)

    # Loss and accuracy for this iteration
    loss_history.append(loss_fn(y, p))
    acc_history.append(accuracy(y, p))
    w_history.append(w.copy())
    b_history.append(b)

    # Gradients of the loss w.r.t. parameters
    error = p - y
    dw = (X.T @ error) / m
    db = np.mean(error)

    # Parameter update:  theta = theta - lr * gradient
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | loss = {loss_history[-1]:.4f} | "
              f"acc = {acc_history[-1]:.2%} | w = {np.round(w, 3)} | b = {b:.3f}")

print(f"\nFinal parameters: w = {np.round(w, 4)}, b = {b:.4f}")
print(f"Final loss: {loss_history[-1]:.4f} | Final accuracy: {acc_history[-1]:.2%}")

# ---------------------------------------------------------------
# TASK 3: Visualize and analyze
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# (a) Loss curve
axes[0].plot(loss_history, color="crimson")
axes[0].set_title("Loss decreasing over iterations")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Cross-entropy loss")
axes[0].grid(alpha=0.3)

# (b) Parameter updates
axes[1].plot(np.array(w_history)[:, 0], label="w1")
axes[1].plot(np.array(w_history)[:, 1], label="w2")
axes[1].plot(b_history, label="bias")
axes[1].set_title("Parameter values during training")
axes[1].set_xlabel("Iteration"); axes[1].legend(); axes[1].grid(alpha=0.3)

# (c) Final decision boundary
axes[2].scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0", alpha=0.7)
axes[2].scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1", alpha=0.7)
xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
axes[2].plot(xs, -(w[0] * xs + b) / w[1], "k--", label="Decision boundary")
axes[2].set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
axes[2].set_title("Learned decision boundary")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent_results.png", dpi=130)
print("\nSaved plot: gradient_descent_results.png")

# Compare learning rates
print("\nEffect of learning rate (final loss after 200 epochs):")
for lr in [0.001, 0.01, 0.1, 1.0]:
    w_t, b_t = np.zeros(2), 0.0
    for _ in range(epochs):
        err = sigmoid(X @ w_t + b_t) - y
        w_t -= lr * (X.T @ err) / m
        b_t -= lr * np.mean(err)
    print(f"  lr = {lr:<6} -> loss = {loss_fn(y, sigmoid(X @ w_t + b_t)):.4f}")
