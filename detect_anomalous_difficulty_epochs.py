import pandas as pd
import numpy as np
from scipy.stats import chi2

# Load block id + time
data = pd.read_csv("combined_times.tsv", sep="\t")

# Ensure proper types
data["time"] = pd.to_datetime(data["time"], errors="coerce")
data = data.dropna(subset=["time", "id"])

# Sort by block height
data = data.sort_values("id").reset_index(drop=True)

# Assign Bitcoin difficulty epoch:
# epoch 0 = blocks 0–2015, epoch 1 = blocks 2016–4031, ...
data["epoch"] = data["id"] // 2016

# Constants
expected_mean = 600
lambda_0 = 1 / expected_mean
alpha = 0.05

def logL(lmbda, intervals):
    n = len(intervals)
    return n * np.log(lmbda) - lmbda * np.sum(intervals)

epochs = data["epoch"].unique()
print(f"Scanning {len(epochs)} difficulty epochs...")

for e in epochs:
    group = data[data["epoch"] == e]

    if len(group) < 2:
        continue  # need at least 2 timestamps

    times = group["time"].sort_values()
    intervals = times.diff().dt.total_seconds().dropna().values

    if len(intervals) < 10:
        continue  # too few blocks in truncated dataset

    # empirical statistics
    emp_mean = np.mean(intervals)
    lambda_mle = 1 / emp_mean

    # log-likelihoods
    LL0 = logL(lambda_0, intervals)
    LL1 = logL(lambda_mle, intervals)
    LR_stat = -2 * (LL0 - LL1)
    p_value = 1 - chi2.cdf(LR_stat, df=1)

    if p_value < alpha:
        start_id = group["id"].min()
        end_id = group["id"].max()
        start_time = group["time"].min().date()
        end_time = group["time"].max().date()
        direction = "GREATER" if emp_mean > expected_mean else "LESS"

        print(f"Epoch {e}: blocks {start_id}–{end_id} ({start_time} → {end_time})")
        print(f"  mean = {emp_mean:.2f}s ({direction})")
        print(f"  p = {p_value:.4g}")
        print()
