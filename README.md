# 🤖 Chatbot Instruction Manual — ML Code Bug Fix

> **To the AI reading this:** You are acting as a **senior ML engineer / code reviewer**.  
> Your job is to read the file `deep_learning_buggy.py`, find the intentional bug,
> fix it, and produce a corrected version of the file. Follow every instruction in this README exactly.

---

## 📁 File Structure

```
project/
├── README.md                  ← You are here (instructions for the AI)
└── deep_learning_buggy.py     ← The Python file you must fix
```

---

## 🧠 What the Code Does

`deep_learning_buggy.py` implements a **multi-layer Deep Neural Network from scratch using NumPy**.  
It is trained on a synthetically generated 3-class **spiral dataset**.

The pipeline includes:

- **Data generation** — `generate_spiral_data()` creates a 2D spiral with 3 arms and Gaussian noise.
- **Preprocessing** — `StandardScaler` normalises inputs; `train_test_split` creates an 80/20 split.
- **Activation functions** — ReLU (hidden layers) and Softmax (output layer).
- **Loss function** — Categorical cross-entropy with one-hot encoding.
- **Neural network class** — `DeepNeuralNetwork` supports:
  - He weight initialisation
  - Mini-batch stochastic gradient descent
  - L2 regularisation
  - Forward and backward propagation
- **Evaluation** — Classification report, confusion matrix, and a saved loss-curve plot.

---

## 🐛 The Bug — What You Must Fix

There is **exactly one bug** in the file. It is located inside the `softmax()` function (around line 45).

### Buggy Code

```python
def softmax(z):
    exp_z = np.exp(z - z.min())                          # ❌ WRONG
    return exp_z / exp_z.sum(axis=1, keepdims=True)
```

### Why It Is Wrong

| Issue | Explanation |
|-------|-------------|
| `z.min()` is a **global scalar** | It subtracts the global minimum from every element across the entire batch — this does NOT prevent numerical overflow for large positive values. |
| Should subtract **row-wise maximum** | The numerically stable softmax requires `z.max(axis=1, keepdims=True)` so each row is shifted independently before exponentiation. |
| Consequence | For inputs with large positive logits, `np.exp(z - z.min())` produces `inf` or `nan`, corrupting the probability distribution and breaking gradient flow silently. |

### Correct Code

```python
def softmax(z):
    exp_z = np.exp(z - z.max(axis=1, keepdims=True))    # ✅ CORRECT
    return exp_z / exp_z.sum(axis=1, keepdims=True)
```

---

## ✅ Your Step-by-Step Task

Follow these steps **in order**:

### Step 1 — Read the file
Read `deep_learning_buggy.py` in full. Understand the complete architecture before making any edits.

### Step 2 — Locate the bug
Find the `softmax()` function and confirm the bug matches the description above.

### Step 3 — Apply the fix
Replace the single incorrect line:

```python
# FROM (buggy):
exp_z = np.exp(z - z.min())

# TO (correct):
exp_z = np.exp(z - z.max(axis=1, keepdims=True))
```

**Do not change anything else in the file.**

### Step 4 — Add a comment above the fix
After applying the fix, add a short inline comment on the corrected line so future readers understand why:

```python
exp_z = np.exp(z - z.max(axis=1, keepdims=True))  # row-wise max for numerical stability
```

### Step 5 — Remove the old bug comment
Delete the original `# BUG:` comment block that was above the wrong line (lines that begin with `# BUG:` or `# This causes`).

### Step 6 — Update the docstring
The `softmax()` function currently has no docstring. Add one:

```python
def softmax(z):
    """
    Numerically stable softmax activation.
    Subtracts the row-wise maximum before exponentiation to prevent overflow.
    
    Parameters
    ----------
    z : np.ndarray, shape (batch_size, n_classes)
    
    Returns
    -------
    np.ndarray — probability distribution over classes, same shape as z.
    """
```

### Step 7 — Save the corrected file
Save the output as `deep_learning_fixed.py` (do **not** overwrite the original buggy file).

### Step 8 — Write a brief fix summary
At the very end of your response (after producing the fixed file), write a plain-English summary structured like this:

```
## Fix Summary

**File edited:** deep_learning_buggy.py  
**Output file:** deep_learning_fixed.py  
**Bug location:** softmax() function, line ~45  
**Root cause:** <one sentence>  
**Fix applied:** <one sentence>  
**Side effects:** None — no other functions or logic were modified.
```

---

## ⛔ Rules & Constraints

- Fix **only** the bug described above. Do not refactor, optimise, or restructure any other part of the code.
- Do **not** add new imports or dependencies.
- Do **not** rename any variables, functions, or classes.
- Do **not** change hyperparameters (`learning_rate`, `layer_dims`, `epochs`, `batch_size`, etc.).
- Preserve all comments that are **not** directly related to the bug.
- The fixed file must be **runnable end-to-end** with `python deep_learning_fixed.py`.

---

## 🔍 How to Verify Your Fix Is Correct

After applying the fix, mentally (or actually) verify these two properties:

1. **Row sums to 1** — for any input matrix `z`, `softmax(z).sum(axis=1)` should equal `[1.0, 1.0, ..., 1.0]`.
2. **No NaN/Inf** — even for extreme inputs like `z = np.array([[1000, 999, 998]])`, the output should be a valid probability vector, not `[nan, nan, nan]`.

---

## 📝 Notes for the AI

- The bug is **subtle** — the code runs without crashing under normal conditions, but produces silently incorrect outputs.
- Only the `softmax()` function is broken. All other functions (`relu`, `relu_derivative`, `cross_entropy_loss`, `DeepNeuralNetwork`) are correct.
- This is a common real-world mistake — `min` vs `max` — that can be hard to spot in a large codebase.

---

*README generated as part of a code-review and bug-fixing exercise.*
