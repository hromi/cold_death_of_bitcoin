"""
paper_time.py

Reproduce the block-time trajectories τ_n for several λ values,
as described in Section "Early-time instability after a shock"
of the manuscript.

Y-axis: estimated block time τ_n (seconds)
X-axis: block index n

This uses the canonical paper-faithful model from paper_model.py.
"""

import matplotlib.pyplot as plt
import numpy as np

from paper_model import ModelParams, simulate_paper_model


def run_time_plot(
    D0: float = 1.0e13,
    lambdas=(0.0, 0.002, 0.005, 0.0015),
    n_blocks=140,
):
    """
    Produce block-time trajectories for several lambda values
    at fixed difficulty (retargeting disabled).
    """

    results = {}

    for lam in lambdas:
        params = ModelParams(
            D0=D0,
            lambda_conf=lam,
            n_blocks=n_blocks,
            apply_retarget=False,     # early-time analysis: fixed difficulty
            shock_fraction=0.4,       # 40% shock
            baseline_price=90_000,
        )
        res = simulate_paper_model(params)
        results[lam] = res

    # ------------------------------------------------------------------
    # Plot: τ_n vs n (block time trajectories)
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))

    for lam in lambdas:
        res = results[lam]
        tau_vals = res.tau

        # Cut off infinite block time (cold death) for early visualization
        tau_plot = np.copy(tau_vals)
        tau_plot = np.where(np.isfinite(tau_plot), tau_plot, np.nan)

        plt.plot(
            tau_plot,
            label=rf"$\lambda = {lam}$",
            linewidth=2,
        )

    plt.axhline(600, color='k', linestyle='--', linewidth=1, label=r"$\tau_0=600\,s$")

    plt.xlabel("Block index $n$")
    plt.ylabel("Block time $\\tau_n$ (s)")
    plt.title("Estimated block-time trajectories for different $\\lambda$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_time_plot()
