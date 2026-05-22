import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Himmelblau's function: f(u, v) = (u^2 + v - 11)^2 + (u + v^2 - 7)^2."""
    u, v = x
    return (u**2 + v - 11) ** 2 + (u + v**2 - 7) ** 2


def grad_f(x):
    """Gradient of Himmelblau's function at (u, v).
    df/du = 4 u (u^2 + v - 11) + 2 (u + v^2 - 7)
    df/dv = 2 (u^2 + v - 11) + 4 v (u + v^2 - 7)"""
    u, v = x
    df_du = 4 * u * (u**2 + v - 11) + 2 * (u + v**2 - 7)
    df_dv = 2 * (u**2 + v - 11) + 4 * v * (u + v**2 - 7)
    return np.array([df_du, df_dv])


def gradient_descent(f, grad_f, eta, u0, v0, max_iter=100) -> tuple[list, list]:
    """Gradient descent on f starting at (u0, v0) with step-size schedule eta(t).
    Returns (path, values) where path[t] = (u_t, v_t) and values[t] = f(path[t])."""
    x = np.array([u0, v0], dtype=float)
    path = [x.copy()]
    values = [f(x)]
    for t in range(max_iter):
        x = x - eta(t) * grad_f(x)
        path.append(x.copy())
        values.append(f(x))
    return path, values


def eta_const(t, c=1e-3) -> float:
    """Constant step size eta = c."""
    return c


def eta_sqrt(t, c=1e-3) -> float:
    """Decreasing step size eta = c / sqrt(t + 1)."""
    return c / np.sqrt(t + 1)


def eta_multistep(t, milestones=[30, 80, 100], c=1e-3, eta_init=1e-3) -> float:
    """Multistep schedule: start at eta_init, multiply by c at every milestone passed."""
    drops = sum(1 for m in milestones if t >= m)
    return eta_init * (c ** drops)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_trajectories(paths_by_start, ax=None):
    """Contour of Himmelblau with each trajectory overlaid."""
    us = np.linspace(-6, 6, 400)
    vs = np.linspace(-6, 6, 400)
    U, V = np.meshgrid(us, vs)
    Z = (U**2 + V - 11) ** 2 + (U + V**2 - 7) ** 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contour(U, V, Z, levels=np.logspace(0, 3.5, 25), cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)

    for name, (path, _) in paths_by_start.items():
        P = np.asarray(path)
        ax.plot(P[:, 0], P[:, 1], "-o", markersize=3, label=name)
        ax.plot(P[0, 0], P[0, 1], "ks")
        ax.plot(P[-1, 0], P[-1, 1], "r*", markersize=10)

    minima = np.array([
        [ 3.000000,  2.000000],
        [-2.805118,  3.131312],
        [-3.779310, -3.283186],
        [ 3.584428, -1.848126],
    ])
    ax.scatter(minima[:, 0], minima[:, 1], marker="x", c="black", s=80,
               label="known minima")

    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Gradient descent trajectories on Himmelblau's function")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_value_curves(paths_by_schedule, ax=None):
    """f(x_t) vs t on a log y-axis for several schedules."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, (_, values) in paths_by_schedule.items():
        ax.plot(values, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("f(x_t)")
    ax.set_title("Step-size schedules from (4, -5)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 2a / 2b / 2c -- 100 steps from (4, -5), three schedules
    # -----------------------------------------------------------------------
    u0, v0 = 4.0, -5.0
    max_iter = 100

    schedules = {
        "constant  (2a)": eta_const,
        "sqrt      (2b)": eta_sqrt,
        "multistep (2c)": eta_multistep,
    }

    paths_by_schedule = {}
    for name, eta in schedules.items():
        path, values = gradient_descent(f, grad_f, eta, u0, v0, max_iter)
        paths_by_schedule[name] = (path, values)
        # min over 1 <= t <= 100 -> exclude the initial point at t = 0
        print(f"{name}:  f(u100,v100) = {values[-1]:.6f}   "
              f"min_{{1..100}} f = {min(values[1:]):.6f}")

    # -----------------------------------------------------------------------
    # 2d -- five starting points with the constant step size
    # -----------------------------------------------------------------------
    starts = {
        "p1": (-4.0, 0.0),
        "p2": ( 0.0, 0.0),
        "p3": ( 4.0, 0.0),
        "p4": ( 0.0, 4.0),
        "p5": ( 5.0, 5.0),
    }

    paths_by_start = {}
    print("\nInitialization sweep (constant step size):")
    for name, (u, v) in starts.items():
        path, values = gradient_descent(f, grad_f, eta_const, u, v, max_iter)
        paths_by_start[name] = (path, values)
        uf, vf = path[-1]
        print(f"{name} -> u: {uf:+.6f}   v: {vf:+.6f}   f_final: {values[-1]:.6e}")

    # Plots
    plot_trajectories(paths_by_start)
    plot_value_curves(paths_by_schedule)
    plt.show()
