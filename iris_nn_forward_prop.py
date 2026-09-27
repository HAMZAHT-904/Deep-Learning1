"""
Project 6: Simple Neural Network Forward Propagation
Dataset: Iris Classification

Task 1: Load and prepare the Iris dataset for classification.
Task 2: Build a simple neural network and perform forward propagation.
Task 3: Apply an activation function and display the predicted output.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)  # for reproducible results

# =========================================================
# TASK 1: Load and prepare the Iris dataset
# =========================================================
iris = load_iris()
X = iris.data                # shape (150, 4) -> 4 features per flower
y = iris.target.reshape(-1, 1)  # shape (150, 1) -> class labels 0,1,2
class_names = iris.target_names

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (mean=0, std=1) -> helps the network process data cleanly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels (needed to compare with softmax output later)
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train)
y_test_oh = encoder.transform(y_test)

print("Task 1 - Data prepared")
print("X_train shape:", X_train.shape, "| y_train_oh shape:", y_train_oh.shape)
print("-" * 60)

# =========================================================
# TASK 2: Build a simple neural network + forward propagation
# =========================================================
# Architecture: 4 (input features) -> 8 (hidden neurons) -> 3 (output classes)

n_input = X_train.shape[1]      # 4 features
n_hidden = 8                    # hidden layer size (chosen)
n_output = y_train_oh.shape[1]  # 3 classes

# Randomly initialize weights and biases (this is NOT trained, just a forward pass demo)
W1 = np.random.randn(n_input, n_hidden) * 0.5
b1 = np.zeros((1, n_hidden))

W2 = np.random.randn(n_hidden, n_output) * 0.5
b2 = np.zeros((1, n_output))


def relu(z):
    """Activation function for the hidden layer."""
    return np.maximum(0, z)


def softmax(z):
    """Activation function for the output layer -> turns scores into probabilities."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X):
    """
    One forward pass through the network:
    Input -> Linear -> ReLU -> Linear -> Softmax -> Output probabilities
    """
    Z1 = np.dot(X, W1) + b1      # linear combination, layer 1
    A1 = relu(Z1)                # TASK 3: activation applied here

    Z2 = np.dot(A1, W2) + b2     # linear combination, layer 2 (output layer)
    A2 = softmax(Z2)             # TASK 3: activation applied here (final prediction)

    return A2


print("Task 2 - Network built: 4 -> 8 (ReLU) -> 3 (Softmax)")
print("-" * 60)

# =========================================================
# TASK 3: Apply activation function & display predicted output
# =========================================================
predictions_prob = forward_propagation(X_test)      # probabilities per class
predicted_classes = np.argmax(predictions_prob, axis=1)  # pick highest probability
true_classes = y_test.flatten()

print("Task 3 - Predictions on test samples (first 10 shown):\n")
print(f"{'Sample':<8}{'Predicted':<15}{'True Label':<15}{'Probabilities'}")
for i in range(10):
    pred_name = class_names[predicted_classes[i]]
    true_name = class_names[true_classes[i]]
    probs = np.round(predictions_prob[i], 3)
    print(f"{i:<8}{pred_name:<15}{true_name:<15}{probs}")

# Simple accuracy note (network is untrained/random weights, so this is just for reference)
accuracy = np.mean(predicted_classes == true_classes)
print("-" * 60)
print(f"Accuracy with RANDOM (untrained) weights: {accuracy:.2%}")
print("Note: This is expected to be low/random since the network has not been")
print("trained yet (no backpropagation) — this project only covers FORWARD propagation.")
