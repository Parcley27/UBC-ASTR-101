"""Automatically fit four sine waves to the Galilean-moon positions in pos.csv.

Unlike ASTR101_Lab2_unmatched.py (sliders + hand-colouring points), this does it
computationally:

  1. Convert pixel offsets from Jupiter to Jupiter radii (same as the lab script).
  2. Multi-start search: each moon is a wave  y = A*sin(2*pi*t/P + phi).
     For a trial set of 4 waves, the points in each image are matched to the
     waves with the Hungarian algorithm (a moon can only appear once per image),
     then each wave is refit by least squares to its matched points. Repeat
     until the matching stops changing; keep the trial with the lowest total
     squared residual.
  3. Print period / amplitude / phase for each moon and plot the result.

Each moon is only searched within a loose period/amplitude band (inner moon =
short period + small amplitude, etc.), which is what stops all four waves from
collapsing onto the same moon. Edit MOON_BOUNDS if you want to change that.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment

CSV = "pos.csv"
OUT_PNG = "moon_fit.png"

# Same constants as the lab script
PIX = 4.289e-3  # pixel size (mm)
AU = 1.496e8  # km
R_JUP = 7.1492e4  # km

# (name, (P_min, P_max) days, (A_min, A_max) Jupiter radii), innermost moon first
MOON_BOUNDS = [
    ("Moon 1 (inner)", (1.0, 2.5), (3.0, 9.0)),
    ("Moon 2", (2.5, 5.0), (7.0, 13.0)),
    ("Moon 3", (5.0, 10.0), (11.0, 19.0)),
    ("Moon 4 (outer)", (10.0, 25.0), (20.0, 32.0)),
]
N_STARTS = 400
MAX_ITERS = 30
SEED = 0


def load_points(path):
    """Return (t, y, frame_idx, t_all): one row per measured moon position."""
    df = pd.read_csv(path)
    t_all = df["YYYYDDD.DDD"].to_numpy(float)
    t_all = t_all - t_all[0]  # days since first image
    convert = PIX / df["FL"].to_numpy(float) * df["Distance"].to_numpy(float) * AU / R_JUP

    t, y, frame = [], [], []
    for i in range(len(df)):
        for m in range(4):
            # MX/MY of -1 means "moon not found in this image"; D can hold junk then
            if df[f"MX{m + 1}"].iloc[i] < 0:
                continue
            d = pd.to_numeric(df[f"D{m + 1}"].iloc[i], errors="coerce")
            if not np.isfinite(d):
                continue
            t.append(t_all[i])
            y.append(d * convert[i])
            frame.append(i)
    return np.array(t), np.array(y), np.array(frame), t_all


def wave(p, t):
    P, A, phi = p
    return A * np.sin(2 * np.pi * t / P + phi)


def assign(params, t, y, frame):
    """Match points to waves within each image (one point per wave). -1 = unmatched."""
    label = np.full(len(t), -1)
    for f in np.unique(frame):
        idx = np.where(frame == f)[0]
        cost = np.array([[(y[i] - wave(p, t[i])) ** 2 for p in params] for i in idx])
        rows, cols = linear_sum_assignment(cost)
        label[idx[rows]] = cols
    return label


def refit(p0, t, y, bounds):
    lo = np.array([bounds[0][0], bounds[1][0], -np.inf])
    hi = np.array([bounds[0][1], bounds[1][1], np.inf])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    return least_squares(lambda p: wave(p, t) - y, p0, bounds=(lo, hi)).x


def total_cost(params, t, y, label):
    return sum(np.sum((y[label == k] - wave(p, t[label == k])) ** 2) for k, p in enumerate(params))


def search(t, y, frame, rng):
    best = (np.inf, None, None)
    for _ in range(N_STARTS):
        params = [
            np.array([rng.uniform(*pb), rng.uniform(*ab), rng.uniform(0, 2 * np.pi)])
            for _, pb, ab in MOON_BOUNDS
        ]
        label = assign(params, t, y, frame)
        for _ in range(MAX_ITERS):
            new_params = []
            for k, (_, pb, ab) in enumerate(MOON_BOUNDS):
                sel = label == k
                if sel.sum() < 4:  # too few points to constrain a wave
                    new_params.append(params[k])
                else:
                    new_params.append(refit(params[k], t[sel], y[sel], (pb, ab)))
            params = new_params
            new_label = assign(params, t, y, frame)
            if np.array_equal(new_label, label):
                break
            label = new_label
        cost = total_cost(params, t, y, label)
        if cost < best[0]:
            best = (cost, params, label)
    return best


def main():
    t, y, frame, t_all = load_points(CSV)
    cost, params, label = search(t, y, frame, np.random.default_rng(SEED))

    rms = np.sqrt(cost / len(t))
    print(f"{len(t)} points, RMS residual {rms:.2f} Jupiter radii\n")
    print(f"{'moon':<16}{'period (d)':>12}{'amplitude (RJ)':>16}{'phase (rad)':>13}{'n pts':>7}")
    for k, ((name, _, _), (P, A, phi)) in enumerate(zip(MOON_BOUNDS, params)):
        print(f"{name:<16}{P:>12.3f}{A:>16.2f}{phi % (2 * np.pi):>13.3f}{(label == k).sum():>7}")
    print("\nModel: y = A * sin(2*pi*t/P + phase), t = days since the first image.")

    colors = ["red", "green", "blue", "purple"]  # same moon colours as the lab script
    tt = np.linspace(t_all.min(), t_all.max(), 1000)

    # Same figure size / axes layout / limits as ASTR101_Lab2_unmatched.py
    fig = plt.figure(figsize=(8, 5), dpi=100)
    curv_ax = plt.axes([0.28, 0.33, 0.65, 0.65])
    curv_ax.scatter(t[label < 0], y[label < 0], c=[[0.5, 0.5, 0.5, 1]] * int((label < 0).sum()))
    for k, (p, c) in enumerate(zip(params, colors)):
        curv_ax.scatter(t[label == k], y[label == k], c=c)
        curv_ax.plot(tt, wave(p, tt), c=c)
    curv_ax.set_ylim(-30, 30)
    curv_ax.set_xlabel("Day")
    curv_ax.set_ylabel("Jupiter Radii")

    # Fitted values, in the empty space left of / below the plot
    fig.text(0.025, 0.25, "Fitted solutions\n(P d, A RJ, phase rad)", fontsize=8, weight="bold", va="top")
    for k, (p, c) in enumerate(zip(params, colors)):
        fig.text(0.025, 0.17 - 0.045 * k,
                 f"M{k + 1}: P={p[0]:.3f}  A={p[1]:.2f}  F={p[2] % (2 * np.pi):.3f}",
                 color=c, fontsize=8, va="top")

    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
