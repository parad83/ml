import numpy as np
import matplotlib.pyplot as plt


def argmin_x1(x):
    """Closed-form minimizer of f along x1.
    Solving df/dx1 = 0 gives x1 = (3 x2 + 2 x3 - 1) / 2."""
    x1, x2, x3 = x
    return (3 * x2 + 2 * x3 - 1) / 2


def argmin_x2(x):
    """Closed-form minimizer of f along x2.
    Solving df/dx2 = 0 gives x2 = (x1 + 2 x3 + 5) / 6."""
    x1, x2, x3 = x
    return (x1 + 2 * x3 + 5) / 6


def argmin_x3(x):
    """Closed-form minimizer of f along x3.
    Solving df/dx3 = 0 gives x3 = (x1 + 3 x2 - 4) / 4."""
    x1, x2, x3 = x
    return (x1 + 3 * x2 - 4) / 4


def f(x):
    """Objective: f(x) = exp(x1 - 3 x2 + 3) + exp(3 x2 - 2 x3 - 2) + exp(2 x3 - x1 + 2)."""
    x1, x2, x3 = x
    return (
        np.exp(x1 - 3 * x2 + 3)
        + np.exp(3 * x2 - 2 * x3 - 2)
        + np.exp(2 * x3 - x1 + 2)
    )


def coordinate_descent(f, argmin, x0, max_iter=100, verbose=False):
    """Run coordinate descent using closed-form per-coordinate minimizers.

    At each outer iteration every coordinate is updated in order, in place
    (Gauss-Seidel style), so a later coordinate sees the freshly updated
    earlier ones. Returns the final iterate and the list of all iterates."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        for j in range(len(x)):
            x[j] = argmin[j](x)
        history.append(x.copy())
        if verbose:
            print(f"iter {i + 1}: x = {x}, f(x) = {f(x)}")

    return x, history


# ---------------------------------------------------------------------------
# Plotting helpers (optional validation)
# ---------------------------------------------------------------------------
def plot_f_history(history, f, ax=None):
    """f(x_t) vs t on a log y-axis -- should drop fast then flatten."""
    values = [f(x) for x in history]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, marker="o")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("f(x_t)")
    ax.set_yscale("log")
    ax.set_title("Convergence of f(x_t)")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_coordinate_history(history, ax=None):
    """Each coordinate x_i vs iteration t -- each should settle to a constant."""
    H = np.asarray(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for i in range(H.shape[1]):
        ax.plot(H[:, i], marker="o", label=f"x{i + 1}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("coordinate value")
    ax.set_title("Coordinates over iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# argmin[i] is the closed-form minimizer for coordinate i.
argmin = [argmin_x1, argmin_x2, argmin_x3]


if __name__ == "__main__":
    x0 = [1.0, 20.0, 5.0]
    x_final, history = coordinate_descent(f, argmin, x0, max_iter=25)

    print(f"x1 = {x_final[0]}")
    print(f"x2 = {x_final[1]}")
    print(f"x3 = {x_final[2]}")
    print(f"f(x_final) = {f(x_final)}")

    for t in range(1, len(history)):
        if np.allclose(history[t], history[t - 1]):
            print(f"converged at iteration t = {t - 1} "
                  f"(history[{t}] matches history[{t - 1}])")
            break
    else:
        print("did not converge within max_iter")

    plot_f_history(history, f)
    plot_coordinate_history(history)
    plt.show()
