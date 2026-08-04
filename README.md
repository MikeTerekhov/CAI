# nn.c

A neural network implemented from scratch in C — no dependencies beyond the standard library and `libm`. It builds a tiny matrix library and uses it to train a single-neuron (logistic regression) model that learns the boolean **AND** function via gradient descent.

## Why

This is a learning exercise: build the matrix operations, forward pass, cost function, and backpropagation by hand in plain C to understand what's actually happening under the hood of a neural network, instead of relying on a framework.

## What it does

- Trains on the AND truth table:

  | x1 | x2 | y |
  |----|----|---|
  | 0  | 0  | 0 |
  | 0  | 1  | 0 |
  | 1  | 0  | 0 |
  | 1  | 1  | 1 |

- Model: `a = sigmoid(x·w + b)`, a single neuron with 2 weights and a bias.
- Loss: mean squared error, `(1/n) * sum((y - a)^2)`.
- Optimization: full-batch gradient descent for 100,000 epochs, printing cost every 10,000 epochs.

## Matrix library

Everything is built on one struct:

```c
typedef struct {
   int rows;
   int cols;
   float *start;
} mat;
```

Operations provided:

| Function | Description |
|---|---|
| `mat_alloc(rows, cols)` | Allocate a matrix |
| `mat_free(m)` | Free a matrix's backing memory |
| `mat_fill(m, x)` | Fill every entry with `x` |
| `mat_print(m, name)` / `MAT_PRINT(m)` | Pretty-print a matrix |
| `mat_add(dst, src)` | `dst += src`, elementwise |
| `mat_diff(dst, src)` | `dst -= src`, elementwise |
| `mat_sq(m)` | Elementwise square |
| `mat_mult_elem(dst, a, b)` | Elementwise (Hadamard) product |
| `mat_mult(dst, a, b)` | Matrix product, `dst = a · b` |
| `mat_mult_scalar(m, x)` | Scale every entry by `x` |
| `mat_copy(dst, src)` | Copy `src` into `dst` |
| `transpose(m)` | Return a newly allocated transpose of `m` |
| `rand_init(m)` | Fill with random floats in `[0, 1)` |
| `sigmoid(m)` | Elementwise sigmoid, in place |
| `sig_grad(m)` | Elementwise sigmoid derivative, given `m` already holds `sigmoid(x)` |

`MAT_AT(m, i, j)` is the indexing macro used throughout (`m.start[i * m.cols + j]`), and matrices are stored **row-major**.

## Build & run

```sh
gcc -Wall -o nn nn.c -lm
./nn
```

## Sample output

```
epoch 0: cost = 0.395773
epoch 10000: cost = 0.201915
...
epoch 90000: cost = 0.018250

final:
w = [
 ...
]
b = [
 ...
]
a1 = [
 0.123970
 0.128517
 0.130853
 0.870318
]
```

The final `a1` (predictions) closely tracks the target `y = [0, 0, 0, 1]`.

## Notes / limitations

- The bias `b` is a `[4, 1]` matrix — one independent bias value per training example rather than a single shared scalar. This is unconventional (a real model would broadcast one scalar bias to every example) and means the model can partly "memorize" per-example offsets rather than learning a general decision boundary. Kept as-is since it doesn't affect learning the AND function on this fixed, tiny dataset.
- No train/test split — this trains and evaluates on the same 4 examples, which is fine for demonstrating gradient descent but isn't a generalization test.
- Single fixed learning rate, no momentum/Adam/etc. — deliberately kept simple.

## License

MIT (or update to whatever you prefer once this is on GitHub).
