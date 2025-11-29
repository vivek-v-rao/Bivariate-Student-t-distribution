"""
Compare conditional variance Var(Y|X=x) for bivariate Student t and normal distributions via simulation,
with optional plots (scatter and uncertainty bands for the t case).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_bivariate_t(nu, cov, n_samples=300_000, seed=1234):
    """Simulate bivariate Student t with df=nu and covariance matrix 'cov'."""
    rng = np.random.default_rng(seed)

    # For multivariate t, Cov = (nu / (nu - 2)) * Σ   =>   Σ = (nu - 2)/nu * Cov
    scale = (nu - 2) / nu * cov

    # Step 1: sample from N(0, scale)
    z = rng.multivariate_normal(mean=np.zeros(2), cov=scale, size=n_samples)

    # Step 2: sample chi-square(df=nu)
    w = rng.chisquare(df=nu, size=n_samples)

    # Step 3: scale mixture representation: T = Z * sqrt(nu / W)
    samples = z * np.sqrt(nu / w)[:, None]
    return samples  # columns: [X, Y]


def simulate_bivariate_normal(cov, n_samples=300_000, seed=5678):
    """Simulate bivariate normal with covariance matrix 'cov'."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n_samples)


def conditional_var_t(x, nu, sigma_x, sigma_y, rho):
    """Theoretical Var(Y | X = x) for bivariate Student t."""
    x = np.asarray(x)
    return ((nu - 2.0) + (x**2) / (sigma_x**2)) / (nu - 1.0) * (sigma_y**2 * (1.0 - rho**2))


def conditional_var_normal(sigma_y, rho):
    """Theoretical Var(Y | X = x) for bivariate normal (constant in x)."""
    return sigma_y**2 * (1.0 - rho**2)


def main():
    # Toggles
    do_plot_scatter = True       # dot plot of (X, Y) for Student t
    do_plot_uncertainty = True   # plot mean and ±1 sd vs x (Student t only)

    # Parameters for the bivariate distributions
    nu = 5.0
    sigma_x = 1.0
    sigma_y = 2.0
    rho = 0.5

    # Simulation controls
    n_samples = 300_000
    h = 0.05  # half-width of conditioning band around each x0

    # Grid of x values at which we compare empirical vs theoretical Var(Y | X = x)
    x0_grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * sigma_x

    # Print parameters used
    print("Parameters used:")
    print(f"  nu        = {nu}")
    print(f"  sigma_x   = {sigma_x}")
    print(f"  sigma_y   = {sigma_y}")
    print(f"  rho       = {rho}")
    print(f"  n_samples = {n_samples}")
    print(f"  band h    = {h}")
    print(f"  x0_grid   = {x0_grid}\n")

    # Target covariance matrix of (X, Y)
    cov = np.array([
        [sigma_x**2,              rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2            ],
    ])

    # -------- BIVARIATE STUDENT t --------
    samples_t = simulate_bivariate_t(nu, cov, n_samples=n_samples, seed=1234)
    x_t = samples_t[:, 0]
    y_t = samples_t[:, 1]

    print("Sample covariance matrix of (X, Y) for Student t:")
    print(np.cov(samples_t.T))

    rows_t = []
    for x0 in x0_grid:
        mask = (x_t > x0 - h) & (x_t < x0 + h)
        y_sel = y_t[mask]
        emp_var = y_sel.var(ddof=1)
        theo_var = conditional_var_t(x0, nu, sigma_x, sigma_y, rho)
        rows_t.append({
            "x0": x0,
            "n_points": y_sel.size,
            "empirical_var": emp_var,
            "theoretical_var": theo_var,
            "ratio_empirical/theoretical": emp_var / theo_var,
        })

    df_t = pd.DataFrame(rows_t)
    print("\nStudent t: comparison of empirical and theoretical Var(Y | X = x):")
    print(df_t.to_string(index=False))

    # -------- BIVARIATE NORMAL (for numeric comparison only) --------
    samples_n = simulate_bivariate_normal(cov, n_samples=n_samples, seed=5678)
    x_n = samples_n[:, 0]
    y_n = samples_n[:, 1]

    print("\nSample covariance matrix of (X, Y) for normal:")
    print(np.cov(samples_n.T))

    theo_var_n = conditional_var_normal(sigma_y, rho)

    rows_n = []
    for x0 in x0_grid:
        mask = (x_n > x0 - h) & (x_n < x0 + h)
        y_sel = y_n[mask]
        emp_var = y_sel.var(ddof=1)
        rows_n.append({
            "x0": x0,
            "n_points": y_sel.size,
            "empirical_var": emp_var,
            "theoretical_var": theo_var_n,
            "ratio_empirical/theoretical": emp_var / theo_var_n,
        })

    df_n = pd.DataFrame(rows_n)
    print("\nNormal: comparison of empirical and theoretical Var(Y | X = x):")
    print(df_n.to_string(index=False))

    # -------- DOT PLOT OF BIVARIATE STUDENT t (optional) --------
    if do_plot_scatter:
        plt.figure()
        plt.scatter(x_t, y_t, s=1, alpha=0.2)
        plt.xlabel("X (Student t)")
        plt.ylabel("Y (Student t)")
        plt.title("Dot plot of bivariate Student t samples")
        plt.axhline(0.0, linewidth=0.5)
        plt.axvline(0.0, linewidth=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()

    # -------- UNCERTAINTY PLOT: mean ± 1 sd vs x (Student t only) --------
    if do_plot_uncertainty:
        # slope of E[Y|X=x]
        beta = rho * sigma_y / sigma_x

        # smooth x grid
        x_line = np.linspace(-3.0 * sigma_x, 3.0 * sigma_x, 201)

        # Student t: mean and sd as functions of x
        mu_t = beta * x_line
        sd_t = np.sqrt(conditional_var_t(x_line, nu, sigma_x, sigma_y, rho))

        plt.figure()
        plt.plot(x_line, mu_t, label="Student t E[Y|X=x]")
        plt.plot(x_line, mu_t + sd_t, linestyle="--",
                 label="Student t E[Y|X=x] + 1 sd")
        plt.plot(x_line, mu_t - sd_t, linestyle="--",
                 label="Student t E[Y|X=x] - 1 sd")
        plt.xlabel("x")
        plt.ylabel("y (mean and ±1 conditional sd)")
        plt.title("Mean and ±1 conditional sd of Y given X=x (Student t)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
