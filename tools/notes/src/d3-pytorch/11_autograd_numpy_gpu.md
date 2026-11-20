---
title: Autograd, NumPy and the GPU
eyebrow: PyTorch · note 11
badge: 🔥
kind: note
blurb: Autograd · Tensors and NumPy · Reproducibility · The GPU · The three common errors
lead: The rest of what makes PyTorch PyTorch. Autograd computes every gradient for you; tensors move to and from NumPy; seeds make "random" repeatable; and `.to(device)` puts the work on the GPU. Then the three errors you will meet most (shape, dtype, device) and how to read them.
run: yes
---
## Autograd: gradients for free

Training needs $\partial L / \partial \theta$ for every parameter (note 06), and backpropagation computes them with the chain rule (note 07). PyTorch's **autograd** does that bookkeeping for you:

1. Mark the tensors you want gradients for with `requires_grad=True` (a model's parameters already are).
2. Compute something from them. As each operation runs, PyTorch records it in a **computation graph**: which tensors went in, which operation, which tensor came out.
3. Call `.backward()` on a **scalar** result. PyTorch walks the graph backwards, applying the chain rule at every node.
4. Read the gradients from each leaf tensor's `.grad`.

```python
import torch

a = torch.tensor(10.0, requires_grad=True)
b = torch.tensor(20.0, requires_grad=True)
F = a * b                 # F = ab, recorded in the graph
F.backward()              # dF/da = b,  dF/db = a
print("dF/da =", a.grad, "   dF/db =", b.grad)
print("F remembers how it was made:", F.grad_fn)
```

$\partial F / \partial a = b = 20$ and $\partial F / \partial b = a = 10$: each input's gradient is the *other* input, exactly the product rule.

A longer chain works the same way: every operation contributes one local derivative, and autograd multiplies them.

```python
x = torch.tensor(2.0, requires_grad=True)
u = x ** 2 + 1            # du/dx = 2x = 4
v = torch.sin(u)          # dv/du = cos(u)
v.backward()
print(f"autograd   dv/dx = {x.grad.item():.6f}")
print(f"chain rule dv/dx = cos(x²+1)·2x = {(torch.cos(torch.tensor(5.0)) * 4).item():.6f}")
```

### Three rules worth knowing

- **Gradients accumulate.** Each `.backward()` *adds* to `.grad`. That is why every training loop calls `optimizer.zero_grad()` (or `x.grad.zero_()`) before the next backward pass.
- **Only scalars can start a backward pass** (call `loss.backward()`, where the loss is one number). For a non-scalar `y`, you would have to pass a vector, `y.backward(v)`; in practice you reduce to a scalar first (`y.sum()`, `y.mean()`).
- **Turn tracking off when you don't need it.** Recording the graph costs memory. Wrap evaluation and inference in `with torch.no_grad():` (or `torch.inference_mode()`), and use `t.detach()` to get a copy that is cut off from the graph (needed before `.numpy()` on a tensor that requires grad).

```python
w = torch.tensor(3.0, requires_grad=True)
for step in range(3):
    loss = (w * 2) ** 2
    loss.backward()
    print(f"after backward {step + 1}: w.grad = {w.grad.item()}")   # 24, 48, 72: it accumulates
w.grad.zero_()
print("after zero_():", w.grad.item())

with torch.no_grad():
    y = w * 2
print("inside no_grad: y.requires_grad =", y.requires_grad)
print("detach:", (w * 2).detach().requires_grad, " → safe to call .numpy():", (w * 2).detach().numpy())
```

:::pq answer="12" type="num" tol="0" tag="Practice"
What will happen when you run the following code? Give the number stored in `x.grad`.

```py
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)
```
@@why
`y = x³`, so autograd computes

$$\frac{dy}{dx} = 3x^2 = 3 \cdot 2^2 = 12.$$

`y` has one element, so `y.backward()` is allowed (a one-element tensor counts as a scalar), and the gradient lands in `x.grad`. The code prints

```text
tensor([12.])
```

(a one-element tensor, because `x` itself was created as `[2.0]`, shape `(1,)`).

**No clue?** Forget PyTorch for a moment: `x.grad` after `y.backward()` is just $dy/dx$ evaluated at the current `x`. Differentiate by hand ($x^3 \to 3x^2$), plug in $x = 2$. Then confirm it in a **+ cell**.
:::

:::pq answer="D" type="single" tag="Practice"
What will be printed?

```py
x = torch.tensor([3.0], requires_grad=True)

y = x * 2
z = y ** 2

z.backward()

print(x.grad)
```
@@
0
@@
12.0
@@
18.0
@@
24.0
@@
Error
@@why
This question is about **Automatic Differentiation (Autograd)** in PyTorch.

Before solving it, let's understand the important functions.

- **`requires_grad=True`** → Tells PyTorch to keep track of all operations on this tensor so that gradients can be computed later.
- **Gradient** → The derivative of the output with respect to a variable. It tells us **how much the output changes when the input changes**.
- **`.backward()`** → Computes the gradients using **Backpropagation** and stores them in the `.grad` attribute of every tensor with `requires_grad=True`.

Let's execute the code step by step.

```py
x = torch.tensor([3.0], requires_grad=True)
```

We create

$$
x=3
$$

and tell PyTorch to track its gradient.

```py
y = x * 2
```

So,

$$
y=2x
$$

Substituting

$$
x=3,
$$

we get

$$
y=6.
$$

```py
z = y ** 2
```

This means

$$
z=y^2.
$$

Substituting

$$
y=6,
$$

we get

$$
z=36.
$$

The computation graph is

```text
x = 3
   │
 ×2
   │
y = 6
   │
Square
   │
z = 36
```

```py
z.backward()
```

Now PyTorch computes $$ \frac{dz}{dx} $$

We use the **Chain Rule**.

Since

$$
z=y^2,
$$

its derivative is

$$
\frac{dz}{dy}=2y.
$$

Since

$$
y=2x,
$$

its derivative is $$ \frac{dy}{dx}=2. $$

Using the Chain Rule, $$ \frac{dz}{dx} = \frac{dz}{dy} \times \frac{dy}{dx}. $$

Substituting the values,

$$
=(2y)\times2
$$

Since

$$
y=6,
$$

$$
=(2\times6)\times2
$$

$$
=12\times2
$$

$$
=24.
$$

PyTorch stores this gradient in

```py
x.grad
```

```py
print(x.grad)
```

prints

```text
tensor([24.])
```

or simply

```text
24.0
```

> **Option A ❌ :** 0

The gradient is not zero.

> **Option B ❌ :** 12.0

This is only

$$
\frac{dz}{dy},
$$

not

$$
\frac{dz}{dx}.
$$

> **Option C ❌ :** 18.0

This value is not obtained from differentiation.

> **Option D ✅ :** 24.0

Using the Chain Rule,

$$
\frac{dz}{dx}=24.
$$

PyTorch stores this in `x.grad`.

✅ **Option D is Correct.**

> **Option E ❌ :** Error

There is no error.

The computation graph is valid, and `backward()` successfully computes the gradient.

**No clue?** Write the chain as a list of steps, `x → y = 2x → z = y²`, put the local derivative next to each arrow (2 and 2y), and multiply them at the current values: y = 6, so dz/dx = 2y · 2 = 24. Autograd does exactly this multiplication. Check it in a **+ cell**.
:::

## Tensors and NumPy

Data often starts in NumPy (pandas, scikit-learn, image libraries), and plotting usually ends there. Two functions convert:

- [`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html): NumPy → tensor.
- [`tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html): tensor → NumPy (CPU tensors only).

Both **share memory**: no copy is made, so changing one changes the other. And NumPy's default float is `float64`, which PyTorch keeps, while PyTorch's default (and what layers expect) is `float32`: convert with `.float()`.

```python
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array.dtype, "→", tensor.dtype, "   convert for a model:", tensor.float().dtype)
array[0] = 100.0
print("shared memory:", tensor[:3])

t = torch.ones(4)
n = t.numpy()
t += 1
print("tensor → numpy, also shared:", n)
```

## Reproducibility

"Random" numbers come from a pseudo-random generator: start it from the same **seed** and it produces the same sequence. Seeding makes experiments repeatable: the same initial weights, the same shuffles, the same results, so a change in the result means a change in your code.

```python
print("unseeded, different every run:", torch.rand(3))

torch.manual_seed(42)
a = torch.rand(3)
torch.manual_seed(42)
b = torch.rand(3)
print("seeded twice with 42:", a, b, "equal:", torch.equal(a, b))
```

`torch.manual_seed` seeds the CPU and every GPU. Set it once at the top of a script; to repeat one particular draw, set it again right before it. (Full determinism on a GPU also needs `torch.use_deterministic_algorithms(True)`, because some fast GPU kernels are not deterministic.)

## The GPU

A CPU has a handful of powerful cores built for sequential, branching work. A GPU has thousands of simple cores built to do the *same* arithmetic on many numbers at once, which is exactly what tensor operations are. **CUDA** is NVIDIA's platform for general computing on its GPUs; PyTorch uses it, together with libraries such as cuBLAS (matrix multiplication) and cuDNN (convolutions).

This computer's GPU, as the driver reports it:

```python
import subprocess, shutil
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                          "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not found: no NVIDIA driver on this machine")
print("PyTorch sees CUDA:", torch.cuda.is_available(), "· CUDA", torch.version.cuda, "· cuDNN", torch.backends.cudnn.version())
```

### Device-agnostic code

Write the device once and use it everywhere, so the same code runs on a laptop CPU or a GPU:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "  GPUs:", torch.cuda.device_count())
if device == "cuda":
    print(f"memory allocated {torch.cuda.memory_allocated() / 1024**2:.1f} MB of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Moving tensors (and models)

`.to(device)` returns a **copy** on that device; reassign to keep it (`t = t.to(device)`). Models move the same way (`model.to(device)`, note 12). `cuda:0` is the first GPU; a second would be `cuda:1`.

```python
t = torch.tensor([1, 2, 3])
t_gpu = t.to(device)
print(t.device, "→", t_gpu.device)

try:
    t_gpu.numpy()                         # NumPy only reads CPU memory
except TypeError as e:
    print("TypeError:", str(e).split(".")[0])
print("back to the CPU first:", t_gpu.cpu().numpy(), "  (t_gpu itself stays on", t_gpu.device, ")")
```

How much faster? A large matrix multiplication on each device (the first GPU call also pays a one-off start-up cost, so it is warmed up first):

```python
import time
def bench(dev, n=3000, reps=5):
    A = torch.rand(n, n, device=dev)
    A @ A
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): A @ A
    if dev == "cuda": torch.cuda.synchronize()       # GPU calls return before the work is done
    return (time.perf_counter() - t0) / reps
cpu = bench("cpu")
print(f"CPU: {cpu * 1000:.0f} ms per 3000×3000 matmul")
if torch.cuda.is_available():
    gpu = bench("cuda")
    print(f"GPU: {gpu * 1000:.1f} ms  →  {cpu / gpu:.0f}× faster")
```

## The three common errors

Almost every PyTorch error you will hit is one of three. Each cell below triggers one, prints the message, and shows the fix.

### Shape

Matrix multiplication needs matching inner dimensions (note 10).

```python
A = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)      # (3, 2)
B = torch.tensor([[7, 10], [8, 11], [9, 12]], dtype=torch.float32)   # (3, 2)
try:
    A @ B
except RuntimeError as e:
    print("RuntimeError:", e)
print(f"fix: transpose B → {tuple(A.shape)} @ {tuple(B.T.shape)} = {tuple((A @ B.T).shape)}")
print(A @ B.T)
print("torch.mm is matmul for 2-D tensors:", torch.equal(torch.mm(A, B.T), A @ B.T))
```

### Datatype

Operations like matmul need both operands in the same dtype.

```python
t16 = torch.arange(10., 100., 10., dtype=torch.float16)
t32 = torch.arange(10., 100., 10.)
try:
    t16 @ t32
except RuntimeError as e:
    print("RuntimeError:", e)
print("fix: convert one →", t16.float() @ t32, "or", t16 @ t32.half())
print("float → int8 truncates toward zero:", torch.tensor([3.9, -2.7, 100.0]).type(torch.int8), "(int8 only holds -128…127)")
```

### Device

```python
cpu_t = torch.arange(10., 100., 10.)
dev_t = torch.arange(10., 100., 10., device=device)
try:
    print(cpu_t @ dev_t, "(no error: both are on the CPU on this machine)")
except RuntimeError as e:
    print("RuntimeError:", e)
print("fix: move one →", cpu_t.to(device) @ dev_t)
```

## Summary: what shape, what dtype, where?

When something fails, print three things about every tensor involved:

| Question | Check | Typical fix |
| :---- | :---- | :---- |
| **What shape?** | `t.shape` | `reshape`, `view`, `.T`, `squeeze` / `unsqueeze`, `permute` |
| **What dtype?** | `t.dtype` | `.float()`, `.long()`, `.to(torch.float32)` |
| **Where?** | `t.device` | `.to(device)` on the data *and* the model |

And the habits that prevent most of them: device-agnostic code with one `device` variable, `float32` data for models, `int64` labels for classification losses, a seed at the top, and `torch.no_grad()` around evaluation.
