import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1.  Data Generation
# ─────────────────────────────────────────────
np.random.seed(42)

def generate_spiral_data(n_samples=1000, n_classes=3, noise=0.2):
    """Generate a 2-D spiral dataset with `n_classes` arms."""
    X, y = [], []
    samples_per_class = n_samples // n_classes

    for cls in range(n_classes):
        t = np.linspace(0, 1, samples_per_class)
        angle = t * 4 * np.pi + (2 * np.pi * cls / n_classes)
        r = t
        x1 = r * np.cos(angle) + np.random.randn(samples_per_class) * noise
        x2 = r * np.sin(angle) + np.random.randn(samples_per_class) * noise
        X.append(np.column_stack([x1, x2]))
        y.append(np.full(samples_per_class, cls))

    return np.vstack(X), np.concatenate(y)


X, y = generate_spiral_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ─────────────────────────────────────────────
#  2.  Activation Functions & Helpers
# ─────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))  # Fixed here
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true, n_classes=3):
    m = y_true.shape[0]
    one_hot = np.zeros((m, n_classes))
    one_hot[np.arange(m), y_true.astype(int)] = 1
    log_probs = -np.log(np.clip(y_pred, 1e-12, 1.0))
    return np.mean(np.sum(log_probs * one_hot, axis=1)), one_hot


# ─────────────────────────────────────────────
#  3.  Neural Network (NumPy, from scratch)
# ─────────────────────────────────────────────
class DeepNeuralNetwork:
    def __init__(self, layer_dims, learning_rate=0.01, l2_lambda=1e-4):
        self.layer_dims    = layer_dims
        self.lr            = learning_rate
        self.l2            = l2_lambda
        self.params        = {}
        self.loss_history  = []
        self._init_weights()

    def _init_weights(self):
        np.random.seed(0)
        for l in range(1, len(self.layer_dims)):
            fan_in  = self.layer_dims[l - 1]
            fan_out = self.layer_dims[l]
            # He initialisation
            self.params[f"W{l}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{l}"] = np.zeros((1, fan_out))

    def forward(self, X):
        cache = {"A0": X}
        L = len(self.layer_dims) - 1

        for l in range(1, L):
            Z = cache[f"A{l-1}"] @ self.params[f"W{l}"] + self.params[f"b{l}"]
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = relu(Z)

        # Output layer
        Z_out = cache[f"A{L-1}"] @ self.params[f"W{L}"] + self.params[f"b{L}"]
        cache[f"Z{L}"] = Z_out
        cache[f"A{L}"] = softmax(Z_out)   # ← buggy softmax called here
        return cache

    def backward(self, cache, y_true):
        grads = {}
        m = y_true.shape[0]
        L = len(self.layer_dims) - 1
        _, one_hot = cross_entropy_loss(cache[f"A{L}"], y_true)

        # Output layer gradient
        dZ = cache[f"A{L}"] - one_hot
        grads[f"dW{L}"] = (cache[f"A{L-1}"].T @ dZ) / m + self.l2 * self.params[f"W{L}"]
        grads[f"db{L}"] = dZ.mean(axis=0, keepdims=True)

        for l in range(L - 1, 0, -1):
            dA = dZ @ self.params[f"W{l+1}"].T
            dZ = dA * relu_derivative(cache[f"Z{l}"])
            grads[f"dW{l}"] = (cache[f"A{l-1}"].T @ dZ) / m + self.l2 * self.params[f"W{l}"]
            grads[f"db{l}"] = dZ.mean(axis=0, keepdims=True)

        return grads

    def update(self, grads):
        for l in range(1, len(self.layer_dims)):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def fit(self, X, y, epochs=500, batch_size=64, verbose=True):
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            # Mini-batch SGD
            indices = np.random.permutation(m)
            X_shuf, y_shuf = X[indices], y[indices]

            epoch_loss = 0.0
            for start in range(0, m, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]

                cache = self.forward(Xb)
                loss, _ = cross_entropy_loss(cache[f"A{len(self.layer_dims)-1}"], yb)
                epoch_loss += loss

                grads = self.backward(cache, yb)
                self.update(grads)

            avg_loss = epoch_loss / (m / batch_size)
            self.loss_history.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:>4}/{epochs}  |  Loss: {avg_loss:.5f}")

    def predict(self, X):
        cache = self.forward(X)
        return np.argmax(cache[f"A{len(self.layer_dims)-1}"], axis=1)


# ─────────────────────────────────────────────
#  4.  Train
# ─────────────────────────────────────────────
architecture = [2, 64, 128, 64, 32, 3]   # input → hidden layers → 3 classes

model = DeepNeuralNetwork(
    layer_dims=architecture,
    learning_rate=0.005,
    l2_lambda=1e-4
)

print("=" * 50)
print("Training Deep Neural Network on Spiral Dataset")
print("=" * 50)
model.fit(X_train, y_train, epochs=300, batch_size=64)


# ─────────────────────────────────────────────
#  5.  Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n" + "=" * 50)
print("Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1", "Class 2"]))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))


# ─────────────────────────────────────────────
#  6.  Plot loss curve
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(model.loss_history, color="steelblue", linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/loss_curve.png", dpi=150)
plt.show()
print("\nLoss curve saved.")

# CodeSentinal: created for you by RuchirAdnaik.