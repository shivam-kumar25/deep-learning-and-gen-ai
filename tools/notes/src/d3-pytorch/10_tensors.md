---
title: Tensors
eyebrow: PyTorch · note 10
badge: 🔥
kind: note
blurb: What PyTorch is · Setup and the GPU · What a tensor is · Creating tensors · Reading values out · Zeros, ones, random, ranges · Operations and matrix multiplication · Aggregation · Reshaping, stacking, squeezing · Indexing and slicing · Datatypes and attributes
lead: The one data structure everything in deep learning is made of. What a tensor is, how to create one, and every operation you will use on them all the time: arithmetic, matrix multiplication, aggregation, reshaping, stacking, squeezing, indexing, datatypes. Every cell runs on a live PyTorch kernel, on the GPU when you have one.
run: yes
---
## What PyTorch is

[PyTorch](https://pytorch.org/) is an open-source deep learning framework, started at Meta AI (then Facebook AI Research) and now run by the PyTorch Foundation. It gives you two things:

1. **NumPy-like arrays that run on a GPU.** The same array code runs on the CPU or, thousands of times in parallel, on an NVIDIA GPU.
2. **Automatic differentiation.** It records the operations you do and computes gradients for you, which is all a training loop needs (note 11).

It is "define by run": the computation graph is built as your Python code executes, so you can use ordinary `if`, `for` and a debugger inside a model. It is the most used framework in deep learning research and is used in production at Meta, Microsoft, Tesla, OpenAI and many others.

## Setup and the GPU

The project's Python environment already has PyTorch with CUDA. Import it and check the version:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
```

### Is there a GPU?

- `torch.cuda.is_available()` is `True` when PyTorch can see a CUDA (NVIDIA) GPU.
- `torch.device("cuda")` / `torch.device("cpu")` name where a tensor lives. Writing `device = "cuda" if torch.cuda.is_available() else "cpu"` once at the top lets the same code run on either.

```python
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ·  {props.total_memory / 1024**3:.1f} GB  ·  compute capability {props.major}.{props.minor}")
```

## What a tensor is

A **tensor** is an n-dimensional array of numbers, all of one type. The number of dimensions is its **rank** (PyTorch calls it `ndim`), and the size along each dimension is its **shape**:

$$\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_n}$$

has rank $n$ and shape $(d_1, d_2, \ldots, d_n)$, and holds $d_1 d_2 \cdots d_n$ numbers.

| Rank | Name | Example | Shape |
| :----: | :---- | :---- | :---- |
| 0 | scalar | a loss value, a learning rate | `torch.Size([])` |
| 1 | vector | one example's features; a bias | `[d]` |
| 2 | matrix | a batch of feature vectors; a weight matrix | `[rows, cols]` |
| 3 | 3-D tensor | one colour image | `[channels, height, width]` |
| 4 | 4-D tensor | a batch of images | `[batch, channels, height, width]` |

```python fig="Tensors of rank 0 to 3. Each new dimension stacks copies of the previous shape: a vector is a row of scalars, a matrix a stack of vectors, a 3-D tensor a stack of matrices."
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
def grid(ax, rows, cols, x0=0, y0=0, fc="#dbeafe", vals=True):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c, y0 + rows - 1 - r), 1, 1, fc=fc, ec="#374151"))
            if vals:
                ax.text(x0 + c + 0.5, y0 + rows - 1 - r + 0.5, str(r * cols + c), ha="center", va="center", fontsize=9)
grid(axes[0], 1, 1); axes[0].set_title("scalar · rank 0\nshape []", fontsize=10)
grid(axes[1], 1, 4); axes[1].set_title("vector · rank 1\nshape [4]", fontsize=10)
grid(axes[2], 3, 4); axes[2].set_title("matrix · rank 2\nshape [3, 4]", fontsize=10)
for d, fc in zip(range(2, -1, -1), ["#fde68a", "#c7d2fe", "#dbeafe"]):
    grid(axes[3], 3, 4, x0=d * 0.6, y0=d * 0.6, fc=fc, vals=(d == 0))
axes[3].set_title("3-D tensor · rank 3\nshape [3, 3, 4]", fontsize=10)
for ax in axes:
    ax.set_aspect("equal"); ax.axis("off"); ax.autoscale_view()
    ax.set_xlim(-0.3, 5.6); ax.set_ylim(-0.3, 4.6)
plt.tight_layout(); plt.show()
```

### Why everything becomes a tensor

A model only does arithmetic, so every input has to become numbers first, and tensors are the container:

- **Images:** a colour image is a rank-3 tensor `[channels, height, width]`. A 224 × 224 RGB image is `[3, 224, 224]`: 3 × 224 × 224 = 150,528 numbers. A batch of 32 is `[32, 3, 224, 224]`.
- **Tables:** a table with 1,000 rows and 12 columns is `[1000, 12]`.
- **Text:** a sentence becomes a sequence of token ids, `[sequence_length]`, then a matrix of embeddings, `[sequence_length, embedding_size]`.
- **Model parameters:** the weights and biases of every layer, and their gradients, are tensors too.

```python fig="A colour image as a tensor: three channels (red, green, blue), each a height × width grid of intensities. The tensor's shape is [3, H, W]."
import numpy as np
import matplotlib.pyplot as plt

H = W = 48
yy, xx = np.mgrid[0:H, 0:W] / H
img = np.stack([np.clip(1.2 - 1.8 * np.hypot(xx - 0.35, yy - 0.4), 0, 1),      # red: a soft disc
                np.clip(xx * 0.9, 0, 1),                                       # green: a left-to-right ramp
                np.clip(1 - yy, 0, 1) * 0.9])                                  # blue: a top-to-bottom ramp
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
axes[0].imshow(img.transpose(1, 2, 0)); axes[0].set_title(f"the image (H = {H}, W = {W})", fontsize=10)
for ax, ch, name, cmap in zip(axes[1:], img, ["red", "green", "blue"], ["Reds", "Greens", "Blues"]):
    ax.imshow(ch, cmap=cmap, vmin=0, vmax=1); ax.set_title(f"channel '{name}': [{H}, {W}]", fontsize=10)
for ax in axes: ax.axis("off")
plt.suptitle(f"tensor shape [3, {H}, {W}] = {3 * H * W:,} numbers", fontsize=10)
plt.tight_layout(); plt.show()
```

Shapes are the thing to watch. Most errors you will hit in PyTorch are shape mismatches, and most of the operations below exist to get tensors into the shape the next operation needs.

## Creating tensors

[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html) is the class; you rarely call it directly. The four ways you will use all the time:

1. **From Python data:** `torch.tensor([...])` copies a number, list or nested list.
2. **From NumPy:** `torch.from_numpy(array)` wraps a NumPy array *without copying* (they share memory).
3. **Filled with a value:** `torch.zeros(shape)`, `torch.ones(shape)`, `torch.full(shape, value)`.
4. **Random:** `torch.rand(shape)` (uniform on [0, 1)), `torch.randn(shape)` (standard normal), `torch.randint(low, high, shape)`.

### A scalar

```python
# Create a scalar tensor (0-dimensional tensor)
# Scalars contain a single numerical value and have no dimensions
# They are fundamental building blocks for more complex tensor operations

scalar = torch.tensor(7)  # Create a scalar tensor with value 7

# Display the scalar tensor
print(f"Scalar tensor: {scalar}")
print(f"Scalar value: {scalar.item()}")  # Extract the Python number
print(f"Data type: {scalar.dtype}")       # Check the data type
print(f"Shape: {scalar.shape}")           # Shape is empty for scalars
print(f"Size: {scalar.size()}")           # Alternative way to check shape

# The scalar tensor object contains metadata beyond just the value:
# - dtype: The data type (default is inferred from input)
# - device: The computational device (CPU or GPU)
# - requires_grad: Whether to track gradients for automatic differentiation

scalar
```

### From Python lists

```python
# Create a simple tensors from Python data structures

# Scalar (0D tensor)
scalar_tensor = torch.tensor(7)

# Vector (1D tensor)
vector_tensor = torch.tensor([1, 2, 3, 4, 5])

# Matrix (2D tensor)
# it will be list of lists, where each inner list represents a row of the matrix
matrix_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Scalar Tensor (0D): {scalar_tensor}")
print(f"Vector Tensor (1D): {vector_tensor}")
print(f"Matrix Tensor (2D): {matrix_tensor}")

# Create a 3D tensor (e.g., a batch of images with 3 color channels)
# it will be a list of matrices, where each matrix represents an image
batch_tensor = torch.tensor([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                             [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                            [[128, 128, 128], [64, 64, 64], [32, 32, 32]]])
print("\nBatch Tensor (3D):")
print(batch_tensor, batch_tensor.shape)
```

### From a NumPy array

`torch.from_numpy` does not copy: the tensor and the array are two views of the same memory, so changing one changes the other. (Use `torch.tensor(array)` when you want an independent copy.)

```python
import numpy as np
import torch

# Create a NumPy array
np_array = np.array([[1, 2, 3], [4, 5, 6]])

# Create a tensor from the NumPy array
tensor_from_np = torch.from_numpy(np_array)

print("NumPy Array:")
print(np_array)
print("dtype of NumPy Array:", np_array.dtype)
print("\nTensor from NumPy Array:")
print(tensor_from_np)
print("dtype of Tensor from NumPy Array:", tensor_from_np.dtype)

# change the array: the tensor changes too, because they share memory
np_array[0, 0] = 99
print("\nAfter modifying the NumPy array:")
print("NumPy Array:")
print(np_array)
print("\nTensor from NumPy Array:")
print(tensor_from_np)
```

### Vectors (rank 1)

**Mathematical Definition**

A **vector** represents a one-dimensional tensor containing multiple numerical values arranged in a specific order. In mathematical notation:
- **Mathematical representation**: **v** ∈ ℝⁿ
- **PyTorch shape**: `torch.Size([n])`
- **Dimensions**: 1

**Properties and Characteristics**

**Vectors possess several important properties:**
- **Ordered sequence**: Elements have specific positions (indices)
- **Directional information**: Can represent direction and magnitude
- **Flexible representation**: Can encode various types of information

**Real-world applications:**
- **Housing features**: `[bedrooms, bathrooms, square_feet]`
- **Word embeddings**: Dense representations of words in NLP
- **Feature vectors**: Extracted features from raw data
- **Probability distributions**: Categorical probability outputs

```python
# Create a vector tensor (1-dimensional tensor)
# Vectors contain multiple values arranged in a single dimension
# They are fundamental for representing sequences, features, and embeddings

vector = torch.tensor([7.0, 7.0])  # Create a vector with two identical elements

print(f"Vector tensor: {vector}")
print(f"Data type: {vector.dtype}")
print(f"Dimension: {vector.dim()}")
print(f"Shape: {vector.shape}")
print(f"Number of elements: {vector.numel()}")  # Total number of elements

# Vector interpretation examples:
# [7, 7] could represent:
# - Coordinates in 2D space (x=7, y=7)
# - Two identical measurements
# - A repeated pattern or signal
# - Feature values for two attributes

# Mathematical properties:
print(f"L2 norm (Euclidean length): {torch.norm(vector).item():.4f}")
print(f"Sum of elements: {torch.sum(vector).item()}")
print(f"Mean of elements: {torch.mean(vector.float()).item():.4f}")

vector
```

### Matrices (rank 2)

**Mathematical Definition:**
A **matrix** represents a two-dimensional tensor organized in rows and columns. In mathematical notation:
- **Mathematical representation**: **M** ∈ ℝᵐˣⁿ
- **PyTorch shape**: `torch.Size([m, n])`
- **Dimensions**: 2

**Properties:**
- **Rows (m)**: First dimension, representing horizontal sequences
- **Columns (n)**: Second dimension, representing vertical sequences
- **Linear transformations**: Matrices can represent linear mappings between vector spaces
- **Data organization**: Natural structure for tabular data and 2D relationships

```python
# Create a matrix tensor (2-dimensional tensor)
# Matrices are fundamental for linear algebra operations in machine learning
# They represent linear transformations, weight matrices, and 2D data structures

MATRIX = torch.tensor([[7.0, 8.0],      # First row: [7, 8]
                       [9.0, 10.0]])    # Second row: [9, 10]

print(f"Matrix tensor:\n{MATRIX}")
print(f"Data type: {MATRIX.dtype}")
print(f"Dimension: {MATRIX.dim()}")  # Should return 2 for a matrix
print(f"Shape: {MATRIX.shape}")  # Returns torch.Size([rows, columns])
print(f"Number of dimensions: {MATRIX.ndim}")
print(f"Total elements: {MATRIX.numel()}")

# Matrix properties and operations:
print(f"\nMatrix properties:")
print(f"Determinant: {torch.det(MATRIX.float()).item():.4f}")  # Requires float type
print(f"Trace (sum of diagonal): {torch.trace(MATRIX).item()}")
print(f"Frobenius norm: {torch.norm(MATRIX).item():.4f}")

# Matrix interpretations:
# - Weight matrix in neural networks (input_features × output_features)
# - Transformation matrix for geometric operations
# - Correlation matrix for feature relationships
# - Adjacency matrix for graph representations

MATRIX
```

### Higher-rank tensors (rank 3 and up)

**Mathematical Definition:**
Higher-order tensors extend beyond matrices to represent multi-dimensional data structures. For a rank-3 tensor:
- **Mathematical representation**: **𝒯** ∈ ℝᵈ¹ˣᵈ²ˣᵈ³
- **PyTorch shape**: `torch.Size([d1, d2, d3])`
- **Dimensions**: 3 or higher

**Applications in Deep Learning:**
- **Batch processing**: `[batch_size, features]` or `[batch_size, channels, height, width]`
- **Sequence modeling**: `[batch_size, sequence_length, feature_dimension]`
- **Computer vision**: `[batch_size, channels, height, width]` for image batches
- **Video processing**: `[batch_size, time_steps, channels, height, width]`

```python
# Create a higher-order tensor (3-dimensional tensor)
# This example creates a rank-3 tensor with shape [1, 3, 3]
# The triple bracket notation [[[...]]] indicates three dimensions

TENSOR = torch.tensor([[[1, 2, 3],     # First 3x3 matrix, row 1
                        [3, 6, 9],     # First 3x3 matrix, row 2
                        [2, 4, 5]]])   # First 3x3 matrix, row 3

print(f"3D Tensor:\n{TENSOR}")
print(f"Data type: {TENSOR.dtype}")
print(f"Dimension: {TENSOR.dim()}")  # Should return 3 for a 3D tensor
print(f"Shape: {TENSOR.shape}")  # torch.Size([1, 3, 3])
print(f"Number of dimensions: {TENSOR.ndim}")
print(f"Total elements: {TENSOR.numel()}")

# Shape interpretation: [1, 3, 3]
# - Dimension 0: 1 "slice" or "channel" (outermost dimension)
# - Dimension 1: 3 rows within each slice
# - Dimension 2: 3 columns within each row

print(f"\nDimensional analysis:")
print(f"Outer dimension (axis 0): {TENSOR.shape[0]} - Number of 3x3 matrices")
print(f"Middle dimension (axis 1): {TENSOR.shape[1]} - Number of rows per matrix")
print(f"Inner dimension (axis 2): {TENSOR.shape[2]} - Number of columns per row")

# Real-world analogy: This could represent:
# - A single grayscale image patch (1 channel, 3x3 pixels)
# - One time step of a 3x3 feature map in a CNN
# - A small kernel/filter in convolutional operations
# - A single sample in a batch of 3x3 matrices

TENSOR
```

The shape is `torch.Size([1, 3, 3])`. Read dimensions from the outside in: the outermost bracket holds **1** item, each of those holds **3** rows, and each row holds **3** numbers. Counting the `[` at the start tells you the rank: `[[[` means 3.

**Note:** You might've noticed the usage of lowercase letters for `scalar` and `vector` and uppercase letters for `MATRIX` and `TENSOR`. This was on purpose. In practice, you'll often see scalars and vectors denoted as lowercase letters such as `y` or `a`. And matrices and tensors denoted as uppercase letters such as `X` or `W`.

You also might notice the names matrix and tensor used interchangeably. This is common. Since in PyTorch you're often dealing with `torch.Tensor`s (hence the tensor name), however, the shape and dimensions of what's inside will dictate what it actually is.

## Reading a value out with `.item()`

`.item()` turns a one-element tensor into a plain Python number. You use it to log a loss (`loss.item()`), to compare a metric in an `if`, or to print a value without the `tensor(...)` wrapper. It only works when the tensor holds **exactly one** element; on anything bigger it raises an error, so index first (`t[0, 1].item()`).

```python
# Extract the Python numerical value from a single-element tensor
# The item() method converts a PyTorch tensor to a Python scalar
# This is essential when you need to use tensor values in standard Python operations

python_value = scalar.item()

print(f"Original tensor: {scalar}")
print(f"Extracted Python value: {python_value}")
print(f"Type of tensor: {type(scalar)}")
print(f"Type of extracted value: {type(python_value)}")

# Practical applications of item():
# 1. Loss monitoring: loss_value = loss_tensor.item()
# 2. Metric computation: accuracy = correct_predictions.item() / total_samples
# 3. Conditional logic: if accuracy_tensor.item() > threshold: ...
# 4. Logging: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Note: Attempting item() on multi-element tensors will raise an error
# For multi-element tensors, use indexing to access specific elements first

python_value
```

## Zeros, ones, random, ranges

Every creation function takes a **shape** (as separate numbers or one tuple) and accepts the same three keyword arguments:

- `dtype=` the number type (default `torch.float32` for these functions; see *Datatypes* below),
- `device=` where the tensor lives (`"cpu"` or `"cuda"`),
- `requires_grad=` whether autograd should track it (note 11).

### Zeros and ones

`torch.zeros(shape)` and `torch.ones(shape)` are the usual starting points: biases start at zero, masks start as all ones, and an accumulator starts at zero before you add into it.

```python
zeros = torch.zeros(size=(3, 4))
ones = torch.ones(3, 4)                       # same thing: the shape can be separate numbers
print(zeros, zeros.dtype, zeros.device, sep="\n")
print(f"\nones: shape {tuple(ones.shape)}, sum {ones.sum().item()} (= number of elements, {ones.numel()})")
print(f"memory: {zeros.element_size()} bytes per element × {zeros.numel()} elements = {zeros.element_size() * zeros.numel()} bytes")
```

### Random tensors

Networks start from random weights: if every weight started equal, every neuron in a layer would compute the same thing and receive the same gradient forever (the *symmetry* problem). Three generators cover almost everything:

| Function | Values | Mean | Std |
| :---- | :---- | :----: | :----: |
| `torch.rand(shape)` | uniform on $[0, 1)$ | 0.5 | $1/\sqrt{12} \approx 0.289$ |
| `torch.randn(shape)` | standard normal | 0 | 1 |
| `torch.randint(low, high, shape)` | integers in $[\text{low}, \text{high})$ | | |

```python
torch.manual_seed(0)                          # same "random" numbers every run (note 11)
u = torch.rand(10_000)
n = torch.randn(10_000)
print(f"rand : min {u.min():.3f}  max {u.max():.3f}  mean {u.mean():.3f}  std {u.std():.3f}   (theory 0.5, 0.289)")
print(f"randn: mean {n.mean():.3f}  std {n.std():.3f}   (theory 0, 1)")
print("randint(0, 10, (2, 5)):\n", torch.randint(0, 10, (2, 5)))

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(u.numpy(), bins=50, color="#2563eb"); axes[0].set_title("torch.rand: flat on [0, 1)")
axes[1].hist(n.numpy(), bins=50, color="#dc2626"); axes[1].set_title("torch.randn: bell curve around 0")
plt.tight_layout(); plt.show()
```

An image-shaped random tensor. Note the order: image files and NumPy use `(height, width, channels)`, while PyTorch models expect `(channels, height, width)`; `.permute(2, 0, 1)` converts.

```python
img_hwc = torch.rand(size=(224, 224, 3))      # how an image file is laid out
img_chw = img_hwc.permute(2, 0, 1)            # how a PyTorch model wants it
mb = img_hwc.numel() * img_hwc.element_size() / 1024**2
print(f"{tuple(img_hwc.shape)} → {tuple(img_chw.shape)}   {img_hwc.numel():,} numbers, {mb:.2f} MB as float32")
```

### Ranges and "like" tensors

| Function | Gives | End included? |
| :---- | :---- | :----: |
| `torch.arange(start, end, step)` | `start, start+step, …` up to `end` | **no** |
| `torch.linspace(start, end, steps)` | `steps` evenly spaced values | **yes** |
| `torch.logspace(start, end, steps, base=10)` | `base**linspace(start, end, steps)` | **yes** |
| `torch.zeros_like(t)`, `torch.ones_like(t)`, `torch.rand_like(t)` | same shape, dtype and device as `t` | |

```python
print("arange(0, 10, 2)     :", torch.arange(0, 10, 2))
print("linspace(0, 1, 5)    :", torch.linspace(0, 1, 5))
print("logspace(0, 3, 4)    :", torch.logspace(0, 3, 4))       # 10⁰ … 10³
r = torch.arange(1, 11)
print("zeros_like(arange)   :", torch.zeros_like(r), r.dtype)   # keeps the integer dtype
```

### Identity, full and diagonal matrices

```python
print("eye(3):\n", torch.eye(3))                       # identity: ones on the diagonal
print("eye(2, 4):\n", torch.eye(2, 4))                 # rectangular: rows, columns
print("full((2, 3), 7):\n", torch.full((2, 3), 7.0))   # every element = 7
v = torch.tensor([5, 22, 13])
print("diag(vector) builds a diagonal matrix:\n", torch.diag(v))
M = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("diag(matrix) reads the diagonal back out:", torch.diag(M))
```

## Manipulating tensors

A network is a long chain of a few operations on tensors: addition, subtraction, element-wise multiplication, division and, above all, matrix multiplication.

### Basic operations

- `Addition` : `torch.add(a, b)` or `a + b`
- `Subtraction` : `torch.sub(a, b)` or `a - b`
- `Multiplication` : `torch.mul(a, b)` or `a * b`
- `Division` : `torch.div(a, b)` or `a / b`
- `Exponentiation` : `torch.pow(a, b)` or `a ** b`
- `Matrix Multiplication` : `torch.matmul(a, b)` or `a @ b`
- `Transposition` : `torch.t(a)` or `a.T`

The arithmetic operators work **element by element**, and a plain number is applied to every element:

```python
# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10
```

```python
# Multiply it by 10
tensor * 10
```

Let's subtract a number and this time we'll reassign the `tensor` variable.

```python
# Subtract and reassign
tensor = tensor - 10
tensor
```

```python
# Add and reassign
tensor = tensor + 10
tensor
```

Each operator has a function form, [`torch.mul()`](https://pytorch.org/docs/stable/generated/torch.mul.html) (alias `torch.multiply`), [`torch.add()`](https://pytorch.org/docs/stable/generated/torch.add.html) and so on. They do the same thing; the operators are shorter.

```python
# Can also use torch functions
torch.multiply(tensor, 10)
```

```python
# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)
```

### Both tensors must be on the same device

A tensor lives on one device, and an operation between two tensors runs on *their* device, so both must be on the same one. PyTorch never moves data for you: CPU and GPU memory are separate, and a silent copy would hide a slow transfer. Move a tensor with `.to(device)` (or `.cuda()` / `.cpu()`), which returns a copy on the new device.

```python
a = torch.ones(2, 2, device=device)       # on the GPU if there is one
b = torch.ones(2, 2)                      # always on the CPU
try:
    print((a + b).device)
except RuntimeError as e:
    print("RuntimeError:", str(e).split("\n")[0])
print("after b.to(device):", (a + b.to(device)).device)
```

On a machine without a GPU both tensors are on the CPU and the addition simply works; on this computer's GPU you see the error.

### Broadcasting: operations between different shapes

`A * B` with different shapes still works when the shapes are **compatible**. PyTorch lines the shapes up from the **right** and compares them dimension by dimension; two sizes are compatible when they are **equal** or one of them is **1** (a missing dimension counts as 1). The size-1 side is then stretched to match, without copying any data.

| A | B | Lined up from the right | Result |
| :---- | :---- | :---- | :---- |
| `(3, 4)` | `(4,)` | `3×4` vs `·×4` | `(3, 4)`: B is reused for every row |
| `(3, 4)` | `(3, 1)` | `3×4` vs `3×1` | `(3, 4)`: B's column is reused for every column |
| `(3, 4)` | `(3,)` | `3×4` vs `·×3` | **error**: 4 vs 3 |
| `(2, 1, 5)` | `(4, 1)` | `2×1×5` vs `·×4×1` | `(2, 4, 5)` |

```python
A = torch.arange(1, 13).reshape(3, 4)
row = torch.tensor([10, 20, 30, 40])          # shape (4,)  → one value per column
col = torch.tensor([[1], [100], [10000]])     # shape (3, 1) → one value per row
print("A * row:\n", A * row)
print("A + col:\n", A + col)
try:
    A * torch.tensor([1, 2, 3])               # shape (3,) does not line up with the last dimension 4
except RuntimeError as e:
    print("RuntimeError:", e)
```

You use broadcasting all the time without noticing: adding a bias vector `b` of shape `(out,)` to a batch of outputs `(batch, out)` is a broadcast.

:::pq answer="C" type="single" tag="Practice"
What will happen when the following PyTorch code is executed? (Assume that the system has GPU installed in it)

```py
import torch

a = torch.ones((2,2), device='cuda')
b = torch.ones((2,2))

c = a + b

print(c.device)
```
@@
It will print `cuda:0` since the result follows the device of `a`.
@@
It will print `cpu` since the result follows the device of `b`.
@@
It will raise a `RuntimeError` because tensors on different devices cannot be directly added.
@@
It will automatically move `b` to GPU before addition.
@@why
Before solving the question, let's understand a few concepts.

- **CPU (Central Processing Unit)** → The default device where PyTorch tensors are created.
- **GPU (CUDA Device)** → A faster processor used for heavy computations like deep learning.
- **Tensor Device** → Every PyTorch tensor belongs to a device (CPU or GPU).

Let's go through the code step by step.

```py
a = torch.ones((2,2), device='cuda')
```

- `torch.ones((2,2))` creates a **2 × 2 tensor filled with 1's**.
- `device='cuda'` places the tensor directly on the **GPU**.

So,

```text
a → GPU (cuda:0)
```

```py
b = torch.ones((2,2))
```

Since no device is specified, PyTorch creates the tensor on the **CPU**.

So,

```text
b → CPU
```

Now the addition is attempted.

```py
c = a + b
```

Here,

- `a` is on the **GPU**.
- `b` is on the **CPU**.

PyTorch **does not allow operations between tensors on different devices** because the CPU and GPU have separate memory.

Therefore, before performing any operation, **both tensors must be on the same device**.

For example, either of these would work:

Move `b` to the GPU:

```py
b = b.to("cuda")
c = a + b
```

or move `a` to the CPU:

```py
a = a.to("cpu")
c = a + b
```

Since the given code does **not** move either tensor, the addition fails.

> **Option A ❌ :** It will print `cuda:0` since the result follows the device of `a`.

The addition never succeeds, so `c` is never created.

> **Option B ❌ :** It will print `cpu` since the result follows the device of `b`.

Again, the addition fails before `c` is created.

> **Option C ✅ :** It will raise a `RuntimeError` because tensors on different devices cannot be directly added.

This is exactly what PyTorch does.

Typical error:

```text
RuntimeError:
Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu.
```

✅ **Option C is Correct.**

> **Option D ❌ :** It will automatically move `b` to GPU before addition.

PyTorch **never moves tensors automatically** between devices.

The programmer must explicitly move tensors using methods like:

```py
.to("cuda")
```

or

```py
.cuda()
```

**No clue?** Ask "where does each tensor live?" for every operand: a tensor created without `device=` is on the CPU. If two operands of one operation give different answers, it is an error, never an automatic move. The first cell under *Both tensors must be on the same device* runs this exact situation.
:::

:::pq answer="B" type="single" tag="Practice"
The following PyTorch code snippet operates on two tensors as follows:

```py
import torch

A = torch.randn(3, 4)
B = torch.randn(4)

C = A * B
```
@@
The operation raises a shape mismatch error because shapes `(3,4)` and `(4)` are incompatible.
@@
Tensor **B** is automatically broadcasted to shape `(3,4)`, and each row of **A** is element-wise multiplied with **B**.
@@
Tensor **B** is broadcasted column-wise to shape `(3,1)`, and each column of **A** is multiplied by a scalar.
@@
The result **C** has shape `(3,1)` since PyTorch collapses unmatched dimensions during element-wise multiplication.
@@why
This question is about **Broadcasting** in PyTorch.

What is Broadcasting?

**Broadcasting** is a feature that allows PyTorch to perform operations on tensors of different shapes **without explicitly copying data**.

PyTorch automatically expands the smaller tensor so that both tensors become compatible for element-wise operations.

Let's execute the code step by step.

```py
A = torch.randn(3, 4)
```

This creates a tensor with **3 rows and 4 columns**.

```text
A (3 × 4)

[[a11 a12 a13 a14]
 [a21 a22 a23 a24]
 [a31 a32 a33 a34]]
```

```py
B = torch.randn(4)
```

This creates a **1-dimensional tensor** with 4 elements.

```text
B

[b1 b2 b3 b4]
```

Shape:

```text
(4,)
```

Now,

```py
C = A * B
```

PyTorch compares the shapes.

```text
A : (3,4)

B :    (4,)
```

The last dimension matches (**4**), so PyTorch automatically broadcasts `B` across every row of `A`.

It behaves as if `B` were expanded to

```text
[[b1 b2 b3 b4]
 [b1 b2 b3 b4]
 [b1 b2 b3 b4]]
```

Notice that PyTorch **does not actually copy the data**. It only treats `B` as if it had this shape during the operation.

The multiplication becomes

```text
[[a11×b1  a12×b2  a13×b3  a14×b4]
 [a21×b1  a22×b2  a23×b3  a24×b4]
 [a31×b1  a32×b2  a33×b3  a34×b4]]
```

Therefore,

```text
Shape of C = (3,4)
```

> **Option A ❌ :** The operation raises a shape mismatch error because shapes `(3,4)` and `(4)` are incompatible.

These shapes are **compatible** because broadcasting applies.

> **Option B ✅ :** Tensor **B** is automatically broadcasted to shape `(3,4)`, and each row of **A** is element-wise multiplied with **B**.

This is exactly how PyTorch broadcasting works.

Each row of `A` is multiplied element-wise by the vector `B`.

✅ **Option B is Correct.**

> **Option C ❌ :** Tensor **B** is broadcasted column-wise to shape `(3,1)`, and each column of **A** is multiplied by a scalar.

PyTorch does **not** reshape `B` into `(3,1)`.

Instead, it broadcasts it across the rows.

> **Option D ❌ :** The result **C** has shape `(3,1)` since PyTorch collapses unmatched dimensions during element-wise multiplication.

PyTorch never collapses dimensions during element-wise multiplication.

The result has the same broadcasted shape,

```text
(3,4)
```

💡 Easy Way to Remember

Suppose

```text
A =
[[1 2 3 4]
 [5 6 7 8]
 [9 10 11 12]]
```

and

```text
B =
[10 20 30 40]
```

Then

```text
A * B =
[[10 40 90 160]
 [50 120 210 320]
 [90 200 330 480]]
```

Notice that **the same vector `B` is multiplied with every row of `A`**.

This is exactly what **broadcasting** does.

**No clue?** Write the two shapes right-aligned, one under the other (`3 4` over `· 4`), and check each column: equal, or one of them is 1 (or missing). If every column passes, the result takes the larger size in each column. Run the broadcasting cell above with `torch.randn(3, 4) * torch.randn(4)` to see the shape.
:::

### Matrix multiplication

Matrix multiplication is the operation deep learning spends most of its time on: every linear layer, every attention head and (rearranged) every convolution is one.

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$ is

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

each entry is the dot product of a row of $A$ with a column of $B$.

#### Where it appears

**Neural Network Forward Pass:**
- Linear layers: **y** = **Wx** + **b**
- Attention mechanisms: **Attention**(**Q**,**K**,**V**) = softmax(**QK**ᵀ/√d)**V**
- Convolutional operations: Implemented as matrix multiplications via im2col

**Training Process:**
- Gradient computation: **∂L/∂W** involves matrix products
- Backpropagation: Chain rule applications through matrix operations

#### The two rules

PyTorch implements matrix multiplication through the [`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html) function, which handles various tensor dimensions automatically.

**Critical Rules for Matrix Multiplication:**

**Rule 1: Inner Dimension Compatibility**
The inner dimensions of the matrices must match for multiplication to be possible:
- **Valid**: (m, n) @ (n, p) → (m, p) ✓
- **Invalid**: (m, n) @ (k, p) where n ≠ k ✗

**Examples:**
- `(3, 2) @ (3, 2)` → **Invalid** (inner dimensions: 2 ≠ 3)
- `(2, 3) @ (3, 2)` → **Valid** (inner dimensions: 3 = 3) → Result: (2, 2)
- `(3, 2) @ (2, 3)` → **Valid** (inner dimensions: 2 = 2) → Result: (3, 3)

**Rule 2: Output Shape Determination**
The resulting matrix has the shape of the outer dimensions:
- (m, **n**) @ (**n**, p) → (m, p)
- The inner dimensions (**n**) are "consumed" during multiplication

**Operator Notation:**
- **Recommended**: `torch.matmul(A, B)` or `A @ B`
- **Alternative**: `torch.mm(A, B)` for 2D matrices only
- **Note**: The `@` operator is the standard Python matrix multiplication symbol (PEP 465)

**Computational Complexity:**
- **Time complexity**: O(mnp) for (m,n) @ (n,p)
- **Space complexity**: O(mp) for the result matrix
- **GPU acceleration**: Highly optimized on modern GPUs using libraries like cuBLAS

```python
import torch
tensor = torch.tensor([1, 2, 3])
tensor.shape
```

The difference between element-wise multiplication and matrix multiplication is the addition of values.

For our `tensor` variable with values `[1, 2, 3]`:

| Operation | Calculation | Code |
| ----- | ----- | ----- |
| **Element-wise multiplication** | `[1*1, 2*2, 3*3]` = `[1, 4, 9]` | `tensor * tensor` |
| **Matrix multiplication** | `[1*1 + 2*2 + 3*3]` = `[14]` | `tensor.matmul(tensor)` |

```python
# Element-wise matrix multiplication
tensor * tensor
```

```python
# For two vectors, matmul is the dot product: 1·1 + 2·2 + 3·3 = 14, a 0-dimensional tensor
matrix_result = torch.matmul(tensor, tensor)
print(matrix_result, matrix_result.shape, "→ as a number:", matrix_result.item())
print("@ is the same operation:", tensor @ tensor)
```

Matrices, and the inner-dimension rule in action. Change the shapes and run it again:

```python
A = torch.rand(2, 3)
B = torch.rand(3, 5)
print("(2, 3) @ (3, 5) →", tuple((A @ B).shape))          # inner 3 = 3: fine, result is the outer (2, 5)
try:
    A @ torch.rand(2, 5)                                  # inner 3 ≠ 2
except RuntimeError as e:
    print("RuntimeError:", e)
print("fix by transposing: (2, 3)ᵀ @ (2, 5) →", tuple((A.T @ torch.rand(2, 5)).shape))

# timing: the GPU's advantage grows with size
import time
for n in (256, 2048):
    X = torch.rand(n, n, device=device)
    X @ X; torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter(); X @ X
    if device.type == "cuda": torch.cuda.synchronize()
    print(f"{n}×{n} @ {n}×{n} on {device}: {(time.perf_counter() - t0) * 1000:.2f} ms")
```

### Aggregation: min, max, sum, mean

An **aggregation** reduces many numbers to fewer: the whole tensor to one number, or one dimension away.

```python
x = torch.arange(0, 100, 10)
print(x, x.dtype)
print(f"min {x.min()}   max {x.max()}   sum {x.sum()}")
try:
    x.mean()                                   # integers: PyTorch refuses to average them
except RuntimeError as e:
    print("RuntimeError:", str(e).split(".")[0])
print(f"mean {x.float().mean()}")              # convert to float first
```

> **Note:** `mean()` (and `std()`, `var()`) need a floating-point tensor; `x.float()` (= `x.type(torch.float32)`) converts. `torch.arange` with whole numbers gives `int64`.

`argmax` / `argmin` give the **position** of the largest / smallest value instead of the value. You will use `argmax` constantly: a classifier outputs one score per class, and `scores.argmax(dim=1)` is the predicted class of each example.

```python
t = torch.tensor([10, 70, 30, 90, 20])
print(f"max {t.max()} at index {t.argmax()},  min {t.min()} at index {t.argmin()}")
scores = torch.tensor([[0.1, 2.3, -0.5], [1.7, 0.2, 0.9]])      # 2 examples × 3 classes
print("predicted class per example:", scores.argmax(dim=1))
```

### Aggregating along one dimension

Pass `dim=` to aggregate along one dimension only. The rule that never fails: **the dimension you name is the one that disappears.**

For a matrix of shape `(rows, cols)`:

- `dim=0` collapses the rows: one result **per column**, shape `(cols,)`.
- `dim=1` collapses the columns: one result **per row**, shape `(rows,)`.
- `keepdim=True` keeps the collapsed dimension with size 1, `(1, cols)` or `(rows, 1)`, which is handy for broadcasting the result back against the original.

```python fig="mean(dim=0) averages down each column (the rows disappear); mean(dim=1) averages across each row (the columns disappear)."
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

X = np.array([[2., 4., 6.], [1., 3., 5.]])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, d in zip(axes, (0, 1)):
    for r in range(2):
        for c in range(3):
            ax.add_patch(Rectangle((c, 1 - r), 1, 1, fc="#dbeafe", ec="#374151"))
            ax.text(c + 0.5, 1.5 - r, f"{X[r, c]:g}", ha="center", va="center", fontsize=12)
    m = X.mean(axis=d)
    if d == 0:
        for c in range(3):
            ax.annotate("", (c + 0.5, -0.35), (c + 0.5, 0.05), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
            ax.add_patch(Rectangle((c, -1.4), 1, 1, fc="#fee2e2", ec="#dc2626"))
            ax.text(c + 0.5, -0.9, f"{m[c]:g}", ha="center", va="center", fontsize=12)
        ax.set_title("x.mean(dim=0) → shape (3,): one per column", fontsize=10)
    else:
        for r in range(2):
            ax.annotate("", (3.4, 1.5 - r), (3.02, 1.5 - r), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
            ax.add_patch(Rectangle((3.5, 1 - r), 1, 1, fc="#fee2e2", ec="#dc2626"))
            ax.text(4.0, 1.5 - r, f"{m[r]:g}", ha="center", va="center", fontsize=12)
        ax.set_title("x.mean(dim=1) → shape (2,): one per row", fontsize=10)
    ax.set_xlim(-0.3, 4.8); ax.set_ylim(-1.6, 2.3); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
```

```python
x = torch.tensor([[2., 4., 6.],
                  [1., 3., 5.]])
print("mean of everything :", x.mean())
print("mean(dim=0)        :", x.mean(dim=0), "  per column")
print("mean(dim=1)        :", x.mean(dim=1), "  per row")
print("mean(dim=1, keepdim=True):\n", x.mean(dim=1, keepdim=True))
print("x - row means (broadcast back):\n", x - x.mean(dim=1, keepdim=True))
```

:::pq answer="A" type="single" tag="Practice"
Given the following PyTorch code:

```py
import torch

x = torch.tensor([[2., 4., 6.],
                  [1., 3., 5.]])

m1 = x.mean().item()
m2 = x.mean(dim=0)
m3 = x.mean(dim=1)

v1 = m2.mean().item()
v2 = m3.mean().item()
```

Which of the following is true?
@@
`m1 = v1 = v2`
@@
`m1 = v1` but not equal to `v2`
@@
`m1 = v2` but not equal to `v1`
@@
`m1`, `v1`, and `v2` all differ
@@why
Let's execute the code step by step.

First, the tensor is

```text
x =
[[2 4 6]
 [1 3 5]]
```

It has

- **2 rows**
- **3 columns**

```py
m1 = x.mean().item()
```

**What does `mean()` do?**

`mean()` computes the **average of all elements** in the tensor.

The six numbers are

```text
2, 4, 6, 1, 3, 5
```

Their sum is

```text
2 + 4 + 6 + 1 + 3 + 5 = 21
```

Number of elements

```text
6
```

Therefore,

$$
m1=\frac{21}{6}=3.5
$$

```py
m2 = x.mean(dim=0)
```

**What does `dim=0` mean?**

`dim=0` means

> Compute the mean **column-wise**.

Take the average of each column.

```text
Column 1 : (2 + 1)/2 = 1.5

Column 2 : (4 + 3)/2 = 3.5

Column 3 : (6 + 5)/2 = 5.5
```

So,

```text
m2 = [1.5, 3.5, 5.5]
```

```py
v1 = m2.mean().item()
```

Now compute the average of these values.

$$
v1=\frac{1.5+3.5+5.5}{3}
=\frac{10.5}{3}
=3.5
$$

Therefore,

```text
v1 = 3.5
```

```py
m3 = x.mean(dim=1)
```

**What does `dim=1` mean?**

`dim=1` means

> Compute the mean **row-wise**.

Take the average of each row.

```text
Row 1 : (2 + 4 + 6)/3 = 4

Row 2 : (1 + 3 + 5)/3 = 3
```

So,

```text
m3 = [4, 3]
```

```py
v2 = m3.mean().item()
```

Now compute the average of the row means.

$$
v2=\frac{4+3}{2}
=\frac{7}{2}
=3.5
$$

We finally have

```text
m1 = 3.5

v1 = 3.5

v2 = 3.5
```

Therefore,

```text
m1 = v1 = v2
```

> **Option A ✅ :** `m1 = v1 = v2`

All three values are equal to **3.5**.

✅ **Option A is Correct.**

> **Option B ❌ :** `m1 = v1` but not equal to `v2`

`v2` is also **3.5**.

> **Option C ❌ :** `m1 = v2` but not equal to `v1`

`v1` is also **3.5**.

> **Option D ❌ :** `m1`, `v1`, and `v2` all differ

They are all equal.

**No clue?** Don't reason about `dim` in the abstract: write the 2 × 3 grid on paper, circle the columns for `dim=0` (the rows disappear) and the rows for `dim=1`, and average. Then notice why all three agree here: every row has the same number of entries and so does every column, so 'average of the column averages' and 'average of the row averages' both weigh each of the six numbers equally. (With an uneven split, such as averaging group means of different sizes, they would not.) You can also just run the code in a cell below and print `m1, v1, v2`.
:::

## Reshaping, stacking, squeezing, permuting

These change a tensor's **shape** without changing its numbers. They exist because every operation has shape rules (matrix multiplication, broadcasting, a layer's expected input), and most of your debugging will be getting a tensor into the right shape.

| Method | What it does |
| :---- | :---- |
| [`x.reshape(shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html) | Same numbers, new shape (the total count must match). Copies only if it has to. |
| [`x.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | Like `reshape`, but always shares memory with `x` (needs a *contiguous* tensor). |
| [`torch.stack(tensors, dim)`](https://pytorch.org/docs/stable/generated/torch.stack.html) | Joins equal-shaped tensors along a **new** dimension. |
| [`torch.cat(tensors, dim)`](https://pytorch.org/docs/stable/generated/torch.cat.html) | Joins tensors along an **existing** dimension. |
| [`x.squeeze(dim)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | Removes dimensions of size 1 (all of them, or just `dim`). |
| [`x.unsqueeze(dim)`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html) | Inserts a dimension of size 1 at position `dim`. |
| [`x.permute(dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | Reorders the dimensions (a view, no copy). |

### reshape and view

```python
x = torch.arange(1., 13.)                      # 12 numbers, shape (12,)
print(x.reshape(3, 4))
print(x.reshape(2, 2, 3).shape, x.reshape(-1, 6).shape)   # -1 = "work this one out"
try:
    x.reshape(5, 3)                            # 15 ≠ 12
except RuntimeError as e:
    print("RuntimeError:", e)
```

`view` never copies: the view and the original are the same memory, so writing through one changes the other.

```python
x = torch.arange(1., 8.)
z = x.view(1, 7)
z[:, 0] = 5                                    # write through the view …
print("z:", z, "\nx:", x)                      # … and x changed too
```

### stack and cat

```python fig="torch.stack makes a new dimension: two (3,) vectors become a (2, 3) matrix with dim=0, or a (3, 2) one with dim=1. torch.cat joins along an existing dimension, so two (3,) vectors become one (6,) vector."
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

a, b = [11, 21, 19], [18, 48, 29]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
def cells(ax, vals, x0, y0, fc, horiz=True):
    for i, v in enumerate(vals):
        x, y = (x0 + i, y0) if horiz else (x0, y0 - i)
        ax.add_patch(Rectangle((x, y), 1, 1, fc=fc, ec="#374151")); ax.text(x + 0.5, y + 0.5, v, ha="center", va="center")
A, B = "#dbeafe", "#fde68a"
cells(axes[0], a, 0, 1, A); cells(axes[0], b, 0, 0, B); axes[0].set_title("stack([a, b], dim=0) → (2, 3)", fontsize=10)
cells(axes[1], a, 0, 2, A, horiz=False); cells(axes[1], b, 1, 2, B, horiz=False); axes[1].set_title("stack([a, b], dim=1) → (3, 2)", fontsize=10)
cells(axes[2], a, 0, 1, A); cells(axes[2], b, 3, 1, B); axes[2].set_title("cat([a, b], dim=0) → (6,)", fontsize=10)
for ax in axes:
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.7, 3.4); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
```

```python
a = torch.tensor([11, 21, 19]); b = torch.tensor([18, 48, 29])
print("stack dim=0:\n", torch.stack([a, b], dim=0))
print("stack dim=1:\n", torch.stack([a, b], dim=1))
print("cat   dim=0:", torch.cat([a, b], dim=0))
M1 = torch.tensor([[11, 21], [18, 19]]); M2 = torch.tensor([[10, 6], [7, 29]])
print("two (2, 2) stacked → ", tuple(torch.stack([M1, M2]).shape), "  stacked on the last dim (dim=-1):\n", torch.stack([M1, M2], dim=-1))
```

`stack` needs every tensor to have **exactly** the same shape; `cat` needs them to match everywhere except along `dim`.

### squeeze and unsqueeze

A **singleton** dimension is one of size 1. They appear constantly: a single image is `(3, H, W)` but a model wants a batch, `(1, 3, H, W)`; a model's output for one example is `(1, 10)` and you want `(10,)`.

```python fig="squeeze() removes every dimension of size 1: (1, 3, 1, 4) becomes (3, 4), the same 12 numbers. unsqueeze(dim) inserts a size-1 dimension: (3, 4) becomes (1, 3, 4) with dim=0 or (3, 1, 4) with dim=1."
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 3))
box = lambda x, y, s, c: ax.text(x, y, s, fontsize=11, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.45", fc=c, ec="#374151"))
arrow = lambda x0, y0, x1, y1, s: (ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#374151")),
                                   ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, s, ha="center", fontsize=9.5, family="monospace"))
box(0.5, 1.0, "(1, 3, 1, 4)", "#fee2e2")
box(3.5, 1.0, "(3, 4)", "#dcfce7")
box(6.6, 1.7, "(1, 3, 4)", "#dbeafe")
box(6.6, 0.3, "(3, 1, 4)", "#dbeafe")
arrow(1.25, 1.0, 2.95, 1.0, "squeeze()")
arrow(4.0, 1.12, 5.95, 1.62, "unsqueeze(0)")
arrow(4.0, 0.88, 5.95, 0.38, "unsqueeze(1)")
ax.text(3.5, -0.45, "the same 12 numbers every time; only the size-1 dimensions change", ha="center", fontsize=9, color="#4b5563")
ax.set_xlim(-0.4, 7.6); ax.set_ylim(-0.7, 2.2); ax.axis("off"); plt.show()
```

```python
t = torch.arange(12).reshape(1, 3, 1, 4)
print(tuple(t.shape), "→ squeeze() →", tuple(t.squeeze().shape))
print("squeeze(0) only:", tuple(t.squeeze(0).shape), "  squeeze(1) (size 3, so nothing happens):", tuple(t.squeeze(1).shape))
s = t.squeeze()
print("unsqueeze(0):", tuple(s.unsqueeze(0).shape), "  unsqueeze(1):", tuple(s.unsqueeze(1).shape), "  unsqueeze(-1):", tuple(s.unsqueeze(-1).shape))
img = torch.rand(3, 28, 28)
print("one image → a batch of one:", tuple(img.unsqueeze(0).shape))
```

`squeeze(dim)` on a dimension that is not size 1 does nothing (no error). `unsqueeze` accepts `dim` from `-(ndim+1)` to `ndim`.

:::pq answer="D" type="single" tag="Practice"
Given the following PyTorch code, what is the shape of `x`?

```py
import torch

a = torch.tensor([[1], [2], [3]])   # Shape: (3,1)
b = torch.tensor([4, 5, 6])         # Shape: (3,)
x = torch.stack([a.squeeze(), b.unsqueeze(1)], dim=1)
```
@@
`(3, 2)`
@@
`(6,)`
@@
`(3, 1, 2)`
@@
The code generates an error due to mismatch in dimensions in the two tensors.
@@why
Work out the shape of each tensor going into `torch.stack`, one call at a time.

```text
a              = [[1], [2], [3]]    shape (3, 1)
a.squeeze()    = [1, 2, 3]          shape (3,)     the size-1 dimension is removed

b              = [4, 5, 6]          shape (3,)
b.unsqueeze(1) = [[4], [5], [6]]    shape (3, 1)   a size-1 dimension is inserted at position 1
```

The two calls swapped the shapes: the first tensor is now `(3,)` and the second `(3, 1)`.

`torch.stack` needs every input to have **exactly the same shape**, because it lays them side by side along a brand-new dimension. `(3,)` and `(3, 1)` hold the same three-ish numbers but are different shapes, so PyTorch raises

```text
RuntimeError: stack expects each tensor to be equal size, but got [3] at entry 0 and [3, 1] at entry 1
```

> **Option A ❌ :** `(3, 2)`

That is what you would get from `torch.stack([a.squeeze(), b], dim=1)`: two `(3,)` vectors stacked along a new dimension 1.

> **Option B ❌ :** `(6,)`

That is `torch.cat([a.squeeze(), b])`: joining along the existing dimension, which `stack` never does.

> **Option C ❌ :** `(3, 1, 2)`

That is `torch.stack([a, b.unsqueeze(1)], dim=2)`: two `(3, 1)` tensors stacked along a new last dimension.

> **Option D ✅ :** The code generates an error due to mismatch in dimensions in the two tensors.

✅ **Option D is Correct.**

**No clue?** Write the shape next to every intermediate tensor, as above. `squeeze` only ever removes 1s and `unsqueeze(k)` only ever inserts a 1 at position k, so the shapes are mechanical; then check the rule of the joining function (`stack`: all equal; `cat`: equal except along `dim`). Running it in a **+ cell** shows the error message too.
:::

### permute

`permute` reorders dimensions. The classic case is images: files and NumPy use `(height, width, channels)`, PyTorch layers want `(channels, height, width)`. It returns a view, so the numbers are shared with the original.

```python
x_hwc = torch.rand(224, 224, 3)
x_chw = x_hwc.permute(2, 0, 1)                # new order: old dim 2, then 0, then 1
print(tuple(x_hwc.shape), "→", tuple(x_chw.shape))
x_hwc[0, 0, 0] = 99.0
print("shared memory:", x_chw[0, 0, 0].item())
```

## Indexing and slicing

Indexing works like nested Python lists and NumPy: one index (or slice) per dimension, separated by commas, **outermost dimension first**.

- `x[i]` picks along dimension 0; `x[i, j]` along dimensions 0 and 1; and so on.
- A slice `start:stop:step` keeps `start` and stops **before** `stop`. Missing parts mean "from the beginning", "to the end", "every element"; `:` alone means "everything in this dimension".
- Negative indices count from the end: `x[-1]` is the last row.
- A sliced tensor is re-indexed from 0.

```python
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print("x[0]      :\n", x[0])
print("x[0][0]   :", x[0][0], "   same as x[0, 0]:", x[0, 0])
print("x[0, 0, 0]:", x[0, 0, 0])
print("x[:, 0]   :", x[:, 0], "   all of dim 0, index 0 of dim 1")
print("x[:, :, 1]:", x[:, :, 1], "   column 1 of every row")
print("x[:, 1, 1]:", x[:, 1, 1], "   x[0, 0, :]:", x[0, 0, :])
```

A 2-D example, with `torch.select` (the function form of indexing one dimension):

```python
x = torch.arange(12).reshape(3, 4)
print(x)
print("rows 1 onwards, columns 0-2:\n", x[1:, :3])
print("every other column:\n", x[:, ::2])
print("last row:", x[-1], "   select(x, dim=1, index=2):", torch.select(x, 1, 2))
```

:::pq answer="B" type="single" tag="Practice"
What will be the output of the following PyTorch code?

```py
import torch

x = torch.arange(12).reshape(3,4)
y = x[1:, :3]

print(y[0, 2])
```
@@
2
@@
6
@@
7
@@
8
@@why
Let's execute the code step by step.

```py
x = torch.arange(12)
```

- `torch.arange(12)` creates a tensor containing numbers from **0 to 11**.

```text
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

```py
x = torch.arange(12).reshape(3,4)
```

- `.reshape(3,4)` rearranges the tensor into **3 rows and 4 columns**.

So, `x` becomes

```text
      Column
       0  1  2  3
      ┌────────────┐
Row 0 │ 0  1  2  3 │
Row 1 │ 4  5  6  7 │
Row 2 │ 8  9 10 11 │
      └────────────┘
```

```py
y = x[1:, :3]
```

**Understanding the slicing**

The syntax is

```py
tensor[rows, columns]
```

**Rows → `1:`**

Means

> Start from **Row 1** and go till the end.

So we select

```text
Row 1
Row 2
```

**Columns → `:3`**

Means

> Start from **Column 0** and stop before **Column 3**.

So we select

```text
Column 0
Column 1
Column 2
```

Therefore,

```text
      Column
       0  1  2
      ┌─────────┐
Row 0 │ 4  5  6 │
Row 1 │ 8  9 10 │
      └─────────┘
```

Notice that after slicing, **indices start again from 0**.

So

```text
y =
[[ 4,  5,  6],
 [ 8,  9, 10]]
```

```py
print(y[0,2])
```

Here,

- `0` → First row of `y`
- `2` → Third column of `y`

Looking at the table,

```text
      0  1  2
     ┌────────┐
0 -> │4  5  6│
1 -> │8  9 10│
     └────────┘
```

The element at

```py
y[0,2]
```

is

```text
6
```

> **Option A ❌ :** 2

The value at `y[0,2]` is not 2.

> **Option B ✅ :** 6

The first row, third column of `y` contains **6**.

✅ **Option B is Correct.**

> **Option C ❌ :** 7

7 is not part of the sliced tensor because the last column was excluded.

> **Option D ❌ :** 8

8 is located at `y[1,0]`, not `y[0,2]`.

**No clue?** Write the tensor out as a grid with row and column numbers, cross out what the slice drops (`1:` drops row 0, `:3` drops column 3), then *renumber* what is left from 0 before applying the second index. Or paste the three lines into a **+ cell** and run them.
:::

## Datatypes

Every tensor has one `dtype`. The ones you meet:

| dtype | Alias | Bits | Used for |
| :---- | :---- | :----: | :---- |
| `torch.float32` | `torch.float` | 32 | **the default** for real numbers: weights, activations, data |
| `torch.float16` / `torch.bfloat16` | `torch.half` | 16 | faster, smaller training on modern GPUs ("mixed precision") |
| `torch.float64` | `torch.double` | 64 | NumPy's default; rarely needed in deep learning |
| `torch.int64` | `torch.long` | 64 | **the default** for whole numbers: class labels, indices |
| `torch.bool` | | 8 | masks, results of comparisons |

`torch.tensor([3.0, 6.0])` gives `float32`, `torch.tensor([3, 6])` gives `int64`, and `torch.from_numpy` keeps NumPy's `float64`. Convert with `x.float()`, `x.long()`, `x.to(torch.float16)` or `x.type(torch.float32)`.

Lower precision means less memory and faster arithmetic but fewer correct digits. Mismatched dtypes are the second most common error after shapes: a layer's `float32` weights cannot multiply a `float64` input, and `CrossEntropyLoss` wants `int64` labels.

```python
f32 = torch.tensor([3.0, 6.0, 9.0])
i64 = torch.tensor([3, 6, 9])
f16 = f32.to(torch.float16)
print(f32.dtype, i64.dtype, f16.dtype, torch.from_numpy(np.array([1.0])).dtype)
print("bytes per element:", f32.element_size(), i64.element_size(), f16.element_size())
print("float16 has about 3 significant digits:", torch.tensor(1 / 3, dtype=torch.float16).item(), "vs", torch.tensor(1 / 3).item())
print("int + float is promoted to float:", (i64 + f32).dtype)
W = torch.rand(3, 3)                          # float32, like a layer's weights
try:
    W @ torch.rand(3, dtype=torch.float64)
except RuntimeError as e:
    print("RuntimeError:", e)
```

## Tensor attributes

The three questions to ask about any tensor while debugging: **what shape, what dtype, what device?**

| Attribute | Gives |
| :---- | :---- |
| `t.shape` (= `t.size()`) | the size of each dimension |
| `t.ndim` (= `t.dim()`) | the number of dimensions (rank) |
| `t.dtype` | the number type |
| `t.device` | `cpu` or `cuda:0` |
| `t.numel()` | the number of elements, the product of the shape |
| `t.element_size()` | bytes per element; × `numel()` = memory used |
| `t.requires_grad` | whether autograd tracks it (note 11) |

```python
t = torch.rand(2, 3, 4, device=device)
print(f"shape {tuple(t.shape)}  ndim {t.ndim}  dtype {t.dtype}  device {t.device}")
print(f"numel {t.numel()} = 2·3·4   memory {t.numel() * t.element_size()} bytes   requires_grad {t.requires_grad}")
print("a scalar has ndim", torch.tensor(7).ndim, "and shape", tuple(torch.tensor(7).shape))
```
