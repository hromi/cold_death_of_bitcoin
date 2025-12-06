"""
Paper-faithful implementation of the model in Sec. II.

Implements:
  - tau_n  from eq. (1):   tau_n = K * D_n / H_n
  - rho_n  from eq. (2):   rho_n = P_n R_BTC / (K D_n)
  - H_n    from eq. (3):   H_n = gamma (rho_n - c_min)_+
  - tau_n  as function of (P_n, D_n) from eq. (4)
  - P_crit from eq. (5)
  - price update eq. (6):  P_{n+1} = P_n exp[-lambda (tau_n/tau_0 - 1)]
  - panic reproduction R_n from eq. (7)
  - retargeting eq. (8):   D_{k+1} = D_k * f_k, f_k clipped in [0.25, 4]
  - cold death condition: rho_n <= c_min ⇒ H_n ≈ 0, tau_n → ∞.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

# ----------------------------------------------------------------------
# Constants from the paper / protocol
# ----------------------------------------------------------------------

K = 2**32          # protocol constant used in PoW (hash target scaling)
N_EPOCH = 2016     # blocks per difficulty epoch
TAU0 = 600.0       # target block time in seconds


@dataclass
class ModelParams:
    # Baseline calibration
    D0: float                  # current difficulty
    P0: float = 90_000.0       # baseline price P_0 in USD
    Pcrit_target: float = 26_000.0  # desired P_crit(D0) in USD
    R_BTC: float = 3.125       # block reward in BTC (adjust as desired)

    # Dynamics / simulation control
    lambda_conf: float = 0.05  # confirmation sensitivity λ
    n_blocks: int = 800        # number of blocks to simulate
    apply_retarget: bool = True

    # Shock setup for early-time experiments (Sec. III)
    shock_fraction: float = 0.4     # s in text; P_0 = (1-s) P_baseline
    baseline_price: float = 90_000  # P_baseline used in the shock

    # Numerical clamps
    exponent_clip: float = 5.0      # clamp for the exponential exponent


def calibrate_gamma_and_cmin(params: ModelParams) -> tuple[float, float]:
    """
    Calibrate c_min and gamma from:
      P_crit(D0) ≈ Pcrit_target
      tau(P0, D0) = tau0

    Equations:
      P_crit(D) = K D c_min / R_BTC
      tau(P,D) = K^2 D^2 / (gamma (P R_BTC - K D c_min))
    """

    D0 = params.D0
    P0 = params.P0
    Pcrit = params.Pcrit_target
    R = params.R_BTC

    # From P_crit(D0) = Pcrit_target
    c_min = Pcrit * R / (K * D0)

    # From tau(P0, D0) = TAU0
    denom = (P0 * R - K * D0 * c_min)
    if denom <= 0:
        raise ValueError("Calibration failed: P0 must be > P_crit(D0).")

    gamma = (K**2 * D0**2) / (TAU0 * denom)

    return c_min, gamma


@dataclass
class SimulationResult:
    price: np.ndarray        # P_n
    hashrate: np.ndarray     # H_n
    difficulty: np.ndarray   # D_n
    tau: np.ndarray          # tau_n
    R: np.ndarray            # R_n
    cold_death_block: Optional[int]


def simulate_paper_model(params: ModelParams) -> SimulationResult:
    """
    Full coupled dynamics of (P_n, tau_n, D_n) iterating equations
    (tau_of_P)–(retarget). Returns arrays of length n_blocks (P,H,D,tau,R).

    Cold death is triggered as soon as rho_n <= c_min.
    """

    c_min, gamma = calibrate_gamma_and_cmin(params)

    N = params.n_blocks

    P = np.zeros(N)
    H = np.zeros(N)
    D = np.zeros(N)
    tau = np.zeros(N)
    Rn = np.zeros(N)

    # Difficulty initial condition
    D[0] = params.D0

    # Shocked initial price: P_0 = (1 - s) P_baseline
    P[0] = (1.0 - params.shock_fraction) * params.baseline_price

    cold_death_block: Optional[int] = None

    for n in range(N):
        # Profitability per unit hashrate eq. (2)
        rho_n = P[n] * params.R_BTC / (K * D[n])

        # Check cold-death condition: rho_n <= c_min
        if rho_n <= c_min:
            H[n] = 0.0
            tau[n] = np.inf
            Rn[n] = params.lambda_conf * (tau[n] / TAU0 - 1.0)
            cold_death_block = n
            # Once miners shut down, we stop: next epoch never completes
            break

        # Supply curve for hashrate eq. (3)
        H[n] = gamma * max(rho_n - c_min, 0.0)

        # Block time from eq. (1) / (4)
        # Using tau_n = K D_n / H_n (equivalent to the composed form)
        tau[n] = K * D[n] / H[n]

        # Panic reproduction number eq. (7)
        Rn[n] = params.lambda_conf * (tau[n] / TAU0 - 1.0)

        # Price update eq. (6): P_{n+1} = P_n e^{-R_n}
        if n < N - 1:
            exponent = -Rn[n]

            # Optional safety clamp to avoid numerical blow-ups
            c = params.exponent_clip
            exponent = float(max(min(exponent, c), -c))

            growth_factor = float(np.exp(exponent))
            P[n + 1] = P[n] * growth_factor

            # We keep a mild clamp to avoid ridiculous prices
            P[n + 1] = max(P[n + 1], 0.0)

            # Difficulty update: default is hold constant
            D[n + 1] = D[n]

            # Difficulty retargeting eq. (8)
            if params.apply_retarget and (n + 1) % N_EPOCH == 0:
                start = n + 1 - N_EPOCH
                T_epoch = float(np.sum(tau[start : n + 1]))
                if T_epoch > 0.0:
                    f_k = (N_EPOCH * TAU0) / T_epoch
                    # clip in [0.25, 4]
                    f_k = max(min(f_k, 4.0), 0.25)
                    D[n + 1] = D[n] * f_k

    return SimulationResult(
        price=P,
        hashrate=H,
        difficulty=D,
        tau=tau,
        R=Rn,
        cold_death_block=cold_death_block,
    )


# ----------------------------------------------------------------------
# Minimal example: early-time instability after a 40% shock, fixed D
# (Sec. "Early-time instability after a shock")
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: choose some reference difficulty D0 (you can plug actual current D)
    D0_example = 1.0e13

    lambdas = [0.0, 0.02, 0.05, 0.1]
    results: Dict[float, SimulationResult] = {}

    for lam in lambdas:
        p = ModelParams(
            D0=D0_example,
            lambda_conf=lam,
            n_blocks=25,          # “over 25 blocks”
            apply_retarget=False  # fixed D_n = D0 for this early-time analysis
        )
        res = simulate_paper_model(p)
        results[lam] = res

    # Plot price trajectories as in Fig. 2(a) of the “early-time instability” section
    plt.figure()
    for lam in lambdas:
        res = results[lam]
        plt.plot(
            res.price,
            label=rf"$\lambda={lam}$",
        )
    plt.xlabel("Block index n")
    plt.ylabel("Price P_n (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()
