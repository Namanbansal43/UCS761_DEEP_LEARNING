# ============================================================
# DL LAB 6 – Backpropagation for Handwritten Digit Classification
# Name: Naman Bansal | Roll: 102303496
# ============================================================
# Neural network implemented from scratch (no PyTorch/TensorFlow)
# Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
# Dataset: MNIST (70,000 handwritten digit images, 28x28 pixels)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)


# ============================================================
# 1. LOAD AND PREPROCESS MNIST
# ============================================================
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixels to [0, 1]
X = X / 255.0

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


def one_hot(labels, num_classes=10):
    """One-hot encode integer labels."""
    oh = np.zeros((labels.shape[0], num_classes))
    oh[np.arange(labels.shape[0]), labels] = 1
    return oh


y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

print(f"Training set : X={X_train.shape}, y={y_train_oh.shape}")
print(f"Test set     : X={X_test.shape},  y={y_test_oh.shape}")


# ============================================================
# 2. VISUALIZE SAMPLE DIGITS
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=12)
    ax.axis("off")
plt.suptitle("Sample MNIST Digits", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()


# ============================================================
# 3. ACTIVATION FUNCTIONS & DERIVATIVES
# ============================================================
def relu(z):
    """ReLU: f(z) = max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    return (z > 0).astype(float)


def softmax(z):
    """Softmax (numerically stable)"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================
# 4. PARAMETER INITIALIZATION (He Initialization)
# ============================================================
def initialize_parameters(layer_dims):
    """He initialization for weights, zeros for biases."""
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(
            layer_dims[l - 1], layer_dims[l]
        ) * np.sqrt(2.0 / layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((1, layer_dims[l]))
    return parameters


# ============================================================
# 5. FORWARD PROPAGATION
# ============================================================
def forward_propagation(X, parameters, num_layers):
    """Forward pass: ReLU for hidden layers, Softmax for output."""
    cache = {"A0": X}
    A = X

    # Hidden layers (ReLU)
    for l in range(1, num_layers):
        Z = A @ parameters[f"W{l}"] + parameters[f"b{l}"]
        A = relu(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A

    # Output layer (Softmax)
    Z_out = A @ parameters[f"W{num_layers}"] + parameters[f"b{num_layers}"]
    A_out = softmax(Z_out)
    cache[f"Z{num_layers}"] = Z_out
    cache[f"A{num_layers}"] = A_out

    return A_out, cache


# ============================================================
# 6. CROSS-ENTROPY LOSS
# ============================================================
def compute_loss(y_true, y_pred):
    """Cross-entropy loss: L = -1/m * sum(y * log(y_hat))"""
    m = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred_clipped)) / m


# ============================================================
# 7. BACKWARD PROPAGATION (CHAIN RULE)
# ============================================================
def backward_propagation(y_true, y_pred, cache, parameters, num_layers):
    """Backward pass: compute gradients via chain rule.
    
    For softmax + cross-entropy: dZ_out = y_hat - y  (simplified)
    For hidden layers with ReLU: dZ_l = dA_l * ReLU'(Z_l)
    """
    m = y_true.shape[0]
    gradients = {}

    # Output layer gradient (softmax + cross-entropy shortcut)
    dZ = (y_pred - y_true) / m

    for l in range(num_layers, 0, -1):
        A_prev = cache[f"A{l-1}"]

        # Weight and bias gradients
        gradients[f"dW{l}"] = A_prev.T @ dZ
        gradients[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

        # Propagate gradient to previous layer
        if l > 1:
            dA = dZ @ parameters[f"W{l}"].T
            dZ = dA * relu_derivative(cache[f"Z{l-1}"])

    return gradients


# ============================================================
# 8. PARAMETER UPDATE (SGD)
# ============================================================
def update_parameters(parameters, gradients, learning_rate, num_layers):
    """SGD update: theta = theta - lr * gradient"""
    for l in range(1, num_layers + 1):
        parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]
    return parameters


# ============================================================
# 9. ACCURACY
# ============================================================
def compute_accuracy(y_true_labels, y_pred_probs):
    """Classification accuracy."""
    predictions = np.argmax(y_pred_probs, axis=1)
    return np.mean(predictions == y_true_labels) * 100


# ============================================================
# 10. TRAINING LOOP (MINI-BATCH SGD)
# ============================================================
# Hyperparameters
layer_dims = [784, 128, 64, 10]
learning_rate = 0.1
epochs = 20
batch_size = 128
num_layers = len(layer_dims) - 1

# Initialize
params = initialize_parameters(layer_dims)

total_params = sum(
    params[f"W{l}"].size + params[f"b{l}"].size
    for l in range(1, len(layer_dims))
)
print(f"\nArchitecture : {' -> '.join(str(d) for d in layer_dims)}")
print(f"Total params : {total_params:,}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size   : {batch_size}")
print(f"Epochs       : {epochs}")
print("\n" + "=" * 70)

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    # Shuffle training data
    perm = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[perm]
    y_shuffled = y_train_oh[perm]

    epoch_loss = 0
    num_batches = 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]

        # Forward
        y_pred, cache = forward_propagation(X_batch, params, num_layers)

        # Loss
        loss = compute_loss(y_batch, y_pred)
        epoch_loss += loss
        num_batches += 1

        # Backward
        grads = backward_propagation(y_batch, y_pred, cache, params, num_layers)

        # Update
        params = update_parameters(params, grads, learning_rate, num_layers)

    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)

    # Evaluate
    y_train_pred, _ = forward_propagation(X_train, params, num_layers)
    train_acc = compute_accuracy(y_train, y_train_pred)
    train_accuracies.append(train_acc)

    y_test_pred, _ = forward_propagation(X_test, params, num_layers)
    test_acc = compute_accuracy(y_test, y_test_pred)
    test_accuracies.append(test_acc)

    print(
        f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
    )

print("=" * 70)
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")


# ============================================================
# 11. PLOT TRAINING RESULTS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, epochs + 1), train_losses, "b-o", linewidth=2, markersize=5)
ax1.set_title("Training Loss (Cross-Entropy)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, epochs + 1), train_accuracies, "b-o", label="Train", linewidth=2, markersize=5)
ax2.plot(range(1, epochs + 1), test_accuracies, "r-s", label="Test", linewidth=2, markersize=5)
ax2.set_title("Classification Accuracy", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mnist_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================
y_test_pred_final, _ = forward_propagation(X_test, params, num_layers)
y_pred_labels = np.argmax(y_test_pred_final, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("True", fontsize=13)
ax.set_xticks(range(10))
ax.set_yticks(range(10))

for i in range(10):
    for j in range(10):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color=color, fontsize=10)

plt.colorbar(im)
plt.tight_layout()
plt.savefig("mnist_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, digits=3))


# ============================================================
# 13. VISUALIZE PREDICTIONS
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
indices = np.random.choice(X_test.shape[0], 10, replace=False)

for idx, ax in zip(indices, axes.flat):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    pred = y_pred_labels[idx]
    true = y_test[idx]
    color = "green" if pred == true else "red"
    ax.set_title(f"Pred: {pred} | True: {true}",
                 fontsize=11, color=color, fontweight="bold")
    ax.axis("off")

plt.suptitle("Predictions on Test Samples\n(Green = Correct, Red = Wrong)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("mnist_predictions.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# 14. MISCLASSIFIED EXAMPLES
# ============================================================
misclassified = np.where(y_pred_labels != y_test)[0]
print(f"\nTotal misclassified: {len(misclassified)} out of {len(y_test)} "
      f"({len(misclassified)/len(y_test)*100:.2f}%)")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for idx, ax in zip(misclassified[:10], axes.flat):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred_labels[idx]} | True: {y_test[idx]}",
                 fontsize=11, color="red", fontweight="bold")
    ax.axis("off")

plt.suptitle("Misclassified Examples", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("mnist_misclassified.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone! All plots saved as PNG files.")
