"""Fully automatic version of Lab 2: images -> moon positions -> sine fits -> power law.

  1. Detect Jupiter and the Galilean moons in each *-SHIFT.JPG: Jupiter is the largest
     saturated blob; moon candidates are difference-of-Gaussians peaks in a wide box. A
     per-image tilted axis through Jupiter picks the moons out from stray stars, and offsets
     are measured along that axis.
  2. Convert pixel offsets to Jupiter radii using each image's focal length and distance
     (from pos.csv, which is in the same order as the images), as in the lab script.
  3. Fit four sine waves y = A*sin(2*pi*t/P + phi) with the same multi-start /
     Hungarian-matching search as fit_moons.py, made robust to false detections, plus a
     period-grid search for the (aliased) inner moon.
  4. Fit the power law P = C * a^x on log-log axes (a = amplitude = semi-major axis).
  5. Compare against the manual results (pos.csv positions, observations.csv fit).

Run from this folder:  python auto_pipeline.py [--no-show]
Outputs: auto_detections.png, auto_moon_fit.png, auto_vs_manual_fit.png, auto_power_law_fit.png, auto_positions.csv
"""
import csv
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment

POS_CSV = "pos.csv"
OBS_CSV = "observations.csv"
IMAGE_GLOB = "*-SHIFT.JPG"

# Same constants as the lab script
PIX = 4.289e-3  # pixel size (mm)
AU = 1.496e8  # km
R_JUP = 7.1492e4  # km

# --- detection settings ---
JUPITER_SAT = 250  # pixels at/above this level (0-255) are Jupiter's saturated disc
BOX_HALF_WIDTH = 700  # px either side of Jupiter searched for moons (wide: outer moon + tilt)
BOX_HALF_HEIGHT = 250  # px above/below Jupiter's row
DOG_SMALL, DOG_LARGE = 1.5, 5.0  # difference-of-Gaussians scales (px): moon-sized blobs only
PEAK_SIZE = 9  # px; peaks closer than this are merged
MIN_SNR = 9.0  # DoG peak height in units of the frame's robust noise
MIN_AXIS_DX = 14  # px; peaks nearer than this along x are Jupiter's glints, not moons
MAX_ELONGATION_RJ = 32.0  # Callisto's orbit is ~26 RJ, so nothing farther than this is a Galilean moon
MAX_TILT = 0.35  # |slope| of the moon axis (dy/dx) searched, ~19 degrees
AXIS_TOL = 12.0  # px a moon may sit off the fitted axis (orbital inclination + centroid error)
AXIS_TOL_FRAC = 0.03  # ...plus this fraction of its distance from Jupiter

# --- curve fit settings (same as fit_moons.py, plus outlier rejection) ---
ALIAS_COST_RATIO = 3.0  # inner-moon period candidates this close to the best fit are treated as aliases
OUTLIER_RJ = 3.0  # a point farther than this from its matched wave is treated as a false detection
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
COLORS = ["red", "green", "blue", "purple"]


# ---------------------------------------------------------------- 1. detection
def find_jupiter(gray):
    """Centre and radius of the largest saturated blob."""
    lab, n = ndimage.label(gray >= JUPITER_SAT)
    sizes = ndimage.sum(np.ones_like(gray), lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean(), np.sqrt(m.sum() / np.pi)


def detect(path, max_offset_px):
    """Find Jupiter and the moons in one image.

    Returns a dict: jupiter (x, y), slope of the moon axis, moons [(x, y, offset_px)]
    with offset_px measured along the axis (negative = left of Jupiter), and every
    candidate peak (for the diagnostic plot).
    """
    gray = np.asarray(Image.open(path).convert("L"), dtype=float)
    jx, jy, _ = find_jupiter(gray)
    x0, y0 = int(jx) - BOX_HALF_WIDTH, int(jy) - BOX_HALF_HEIGHT
    box = gray[max(y0, 0):y0 + 2 * BOX_HALF_HEIGHT, max(x0, 0):x0 + 2 * BOX_HALF_WIDTH]
    x0, y0 = max(x0, 0), max(y0, 0)

    # A difference of Gaussians keeps moon-sized blobs and removes sky level, sky gradients
    # (e.g. the twilight sky on Jul 13) and most of Jupiter's smooth halo.
    dog = ndimage.gaussian_filter(box, DOG_SMALL) - ndimage.gaussian_filter(box, DOG_LARGE)
    noise = 1.4826 * np.median(np.abs(dog - np.median(dog)))
    snr = dog / noise
    is_peak = (dog == ndimage.maximum_filter(dog, size=PEAK_SIZE)) & (snr > MIN_SNR)

    cands = []  # (dx, dy, snr) relative to Jupiter, sub-pixel centroid from the DoG weights
    for py, px in zip(*np.nonzero(is_peak)):
        ys, xs = slice(max(py - 3, 0), py + 4), slice(max(px - 3, 0), px + 4)
        w = dog[ys, xs].clip(0)
        gy, gx = np.mgrid[ys, xs]
        cands.append(((w * gx).sum() / w.sum() + x0 - jx, (w * gy).sum() / w.sum() + y0 - jy, snr[py, px]))
    cands = [c for c in cands if MIN_AXIS_DX <= abs(c[0]) <= max_offset_px]

    # Per-image moon axis: the line through Jupiter (dy = slope*dx) that explains the most
    # peak brightness. This both rejects stray stars and accounts for the camera tilt.
    def members(slope):
        return [c for c in cands if abs(c[1] - slope * c[0]) <= AXIS_TOL + AXIS_TOL_FRAC * abs(c[0])]

    slopes = np.linspace(-MAX_TILT, MAX_TILT, 141)
    scores = [sum(np.log(c[2]) for c in sorted(members(s), key=lambda c: -c[2])[:4]) for s in slopes]
    slope = slopes[int(np.argmax(scores))]
    mem = sorted(members(slope), key=lambda c: -c[2])[:4]
    if len(mem) >= 2:  # refine the slope with a least-squares line through Jupiter
        dx, dy = np.array([c[0] for c in mem]), np.array([c[1] for c in mem])
        slope = (dx * dy).sum() / (dx * dx).sum()

    u = np.array([1.0, slope]) / np.hypot(1.0, slope)
    moons = sorted((c[0] * u[0] + c[1] * u[1], jx + c[0], jy + c[1]) for c in mem)
    return {"jupiter": (jx, jy), "slope": slope, "moons": [(x, y, d) for d, x, y in moons],
            "cands": [(jx + c[0], jy + c[1]) for c in cands]}


def load_metadata(path):
    """Per-image time, focal length, distance (AU) from pos.csv, in image order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["YYYYDDD.DDD"]) for r in rows])
    return t - t[0], np.array([float(r["FL"]) for r in rows]), np.array([float(r["Distance"]) for r in rows])


def measure_all():
    files = sorted(glob.glob(IMAGE_GLOB), key=lambda f: (f[2:4], f[0:2]))  # by month, day
    t_all, fl, dist = load_metadata(POS_CSV)
    if len(files) != len(t_all):
        sys.exit(f"{len(files)} images but {len(t_all)} rows in {POS_CSV}")

    scale = PIX / fl * dist * AU / R_JUP  # Jupiter radii per pixel, per image
    results = []
    for f, sc in zip(files, scale):
        r = detect(f, MAX_ELONGATION_RJ / sc)
        results.append(r)
        print(f"{f}: Jupiter at ({r['jupiter'][0]:.0f},{r['jupiter'][1]:.0f}), "
              f"axis tilt {np.degrees(np.arctan(r['slope'])):+.1f} deg, {len(r['moons'])} moons found")

    t, y, frame = [], [], []
    for i, r in enumerate(results):
        for _, _, d in r["moons"]:
            t.append(t_all[i])
            y.append(d * scale[i])
            frame.append(i)
    return files, results, t_all, np.array(t), np.array(y), np.array(frame)


# ---------------------------------------------------------------- 2. sine fits
def wave(p, t):
    P, A, phi = p
    return A * np.sin(2 * np.pi * t / P + phi)


def assign(params, t, y, frame):
    label = np.full(len(t), -1)
    for f in np.unique(frame):
        idx = np.where(frame == f)[0]
        cost = np.array([[(y[i] - wave(p, t[i])) ** 2 for p in params] for i in idx])
        rows, cols = linear_sum_assignment(cost)
        label[idx[rows]] = cols
    for i in np.nonzero(label >= 0)[0]:  # reject false detections instead of letting them drag the fit
        if abs(y[i] - wave(params[label[i]], t[i])) > OUTLIER_RJ:
            label[i] = -1
    return label


def refit(p0, t, y, bounds):
    lo = np.array([bounds[0][0], bounds[1][0], -np.inf])
    hi = np.array([bounds[0][1], bounds[1][1], np.inf])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    return least_squares(lambda p: wave(p, t) - y, p0, bounds=(lo, hi))


def total_cost(params, t, y, label):
    """Sum of squared residuals; every rejected point costs the full OUTLIER_RJ**2 penalty."""
    cost = OUTLIER_RJ ** 2 * np.sum(label < 0)
    return cost + sum(np.sum((y[label == k] - wave(p, t[label == k])) ** 2) for k, p in enumerate(params))


def refine_inner(params, label, t, y, frame, cost):
    """Re-search the inner moon on a fine period grid.

    With roughly daily sampling a fast orbit is badly aliased, so random starts can settle
    on the wrong period. The points not claimed by moons 2-4 are mostly the inner moon, so
    scan its period range, solve for amplitude and phase linearly at each trial period
    (robust re-weighting so stray points do not dominate), and keep the period that explains
    the most points. Accepted only if the total cost improves.
    """
    left = (label < 0) | (label == 0)
    tl, yl = t[left], y[left]
    (_, (pmin, pmax), (amin, amax)) = MOON_BOUNDS[0]
    periods = np.arange(pmin, pmax, 0.002)
    costs, sols = [], []
    for P in periods:
        design = np.column_stack([np.sin(2 * np.pi * tl / P), np.cos(2 * np.pi * tl / P)])
        w = np.ones(len(tl))
        for _ in range(5):
            coef = np.linalg.lstsq(design * w[:, None], yl * w, rcond=None)[0]
            r = yl - design @ coef
            w = 1 / (1 + (r / 1.0) ** 2)
        amp = np.hypot(*coef)
        costs.append(np.sum(np.minimum(r ** 2, OUTLIER_RJ ** 2)) if amin <= amp <= amax else np.inf)
        sols.append(np.array([P, amp, np.arctan2(coef[1], coef[0])]))
    costs = np.array(costs)
    minima = [i for i in range(1, len(costs) - 1) if costs[i] < costs[i - 1] and costs[i] <= costs[i + 1]]
    if not minima:
        return cost, params, label
    best = min(minima, key=lambda i: costs[i])

    # Aliasing: ~daily sampling cannot tell a period P from 1/(1 - 1/P). Among the minima that
    # are nearly as good as the best, prefer the one closest to the period a power law through
    # moons 2-4 predicts at the inner moon's amplitude (Kepler's third law as a physical prior).
    x, b = np.polyfit(np.log([p[1] for p in params[1:]]), np.log([p[0] for p in params[1:]]), 1)
    near_best = [i for i in minima if costs[i] <= ALIAS_COST_RATIO * costs[best]]
    pick = min(near_best, key=lambda i: abs(np.log(sols[i][0]) - (b + x * np.log(sols[i][1]))))
    if pick != best:
        print(f"Inner moon: data alone favour the alias P = {sols[best][0]:.3f} d; "
              f"using P = {sols[pick][0]:.3f} d, the minimum closest to the outer-moon power law.")
    best = (costs[pick], sols[pick])
    new = list(params)
    new[0] = best[1]
    for _ in range(MAX_ITERS):  # polish all four with the usual match/refit loop
        new_label = assign(new, t, y, frame)
        for k, (_, pb, ab) in enumerate(MOON_BOUNDS):
            sel = new_label == k
            if sel.sum() >= 4:
                new[k] = refit(new[k], t[sel], y[sel], (pb, ab)).x
        if np.array_equal(new_label, assign(new, t, y, frame)):
            break
    new_label = assign(new, t, y, frame)
    new_cost = total_cost(new, t, y, new_label)
    return (new_cost, new, new_label) if new_cost < cost else (cost, params, label)


def fit_waves(t, y, frame, rng):
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
                new_params.append(params[k] if sel.sum() < 4 else refit(params[k], t[sel], y[sel], (pb, ab)).x)
            params = new_params
            new_label = assign(params, t, y, frame)
            if np.array_equal(new_label, label):
                break
            label = new_label
        cost = total_cost(params, t, y, label)
        if cost < best[0]:
            best = (cost, params, label)
    cost, params, label = best
    cost, params, label = refine_inner(params, label, t, y, frame, cost)

    # 1-sigma uncertainties on P and A from the least-squares Jacobian
    errs = []
    for k, (_, pb, ab) in enumerate(MOON_BOUNDS):
        sel = label == k
        res = refit(params[k], t[sel], y[sel], (pb, ab))
        dof = max(sel.sum() - 3, 1)
        cov = np.linalg.pinv(res.jac.T @ res.jac) * (2 * res.cost / dof)
        errs.append(np.sqrt(np.diag(cov)))
    return cost, params, label, errs


# ---------------------------------------------------------------- 3. power law
def power_law(period, axis):
    """Fit log10 P = log10 C + x log10 a. Returns x, C, intercept and 1-sigma errors on x."""
    lx, ly = np.log10(axis), np.log10(period)
    if len(lx) > 3:  # polyfit can only estimate a covariance with more than 3 points
        (x, b), cov = np.polyfit(lx, ly, 1, cov=True)
        x_err = np.sqrt(cov[0, 0])
    else:
        x, b = np.polyfit(lx, ly, 1)
        x_err = np.nan
    return x, 10 ** b, b, x_err, lx, ly


def manual_power_law():
    d = np.genfromtxt(OBS_CSV, delimiter=",", skip_header=1)
    return power_law(d[:, 1], d[:, 2])


def manual_waves():
    """Your hand-fitted (P, A) from observations.csv, with phases solved against pos.csv.

    The manual phases were slider values that were not saved, so only the phase is fitted
    here (period and amplitude stay at your values), to your hand-measured positions.
    Returns the waves ordered innermost moon first, plus the manual points (t, y).
    """
    from fit_moons import load_points  # same pixel -> Jupiter-radii conversion on pos.csv

    t, y, frame, _ = load_points(POS_CSV)
    obs = np.genfromtxt(OBS_CSV, delimiter=",", skip_header=1)
    pa = sorted((row[1], row[2]) for row in obs)  # (period, amplitude), shortest period first
    phases = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def phase_cost(P, A, phi):  # per frame, the closest measured point to the wave
        return sum(min(np.min((y[frame == f] - wave((P, A, phi), t[frame == f])) ** 2), OUTLIER_RJ ** 2)
                   for f in np.unique(frame))

    params = [np.array([P, A, phases[np.argmin([phase_cost(P, A, p) for p in phases])]]) for P, A in pa]
    for _ in range(MAX_ITERS):  # match points to waves, then re-solve each phase
        label = assign(params, t, y, frame)
        new = []
        for k, (P, A, phi) in enumerate(params):
            sel = label == k
            cost = [np.sum((y[sel] - wave((P, A, p), t[sel])) ** 2) for p in phases]
            new.append(np.array([P, A, phases[np.argmin(cost)]]))
        if all(np.allclose(a, b) for a, b in zip(new, params)):
            break
        params = new
    return params, t, y, label


# ---------------------------------------------------------------- main
def main():
    show = "--no-show" not in sys.argv
    files, results, t_all, t, y, frame = measure_all()

    with open("auto_positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "day", "offset_RJ"])
        for fi, ti, yi in zip(frame, t, y):
            w.writerow([files[fi], f"{ti:.4f}", f"{yi:.3f}"])

    # detection sanity check: every candidate peak (red) and the accepted moons (yellow)
    fig, axes = plt.subplots(4, 4, figsize=(22, 9))
    for ax, f, r in zip(axes.flat, files, results):
        img = np.asarray(Image.open(f).convert("L"), dtype=float)
        jx, jy = r["jupiter"]
        x0, x1 = int(jx) - BOX_HALF_WIDTH, int(jx) + BOX_HALF_WIDTH
        y0, y1 = int(jy) - BOX_HALF_HEIGHT // 2, int(jy) + BOX_HALF_HEIGHT // 2
        crop = img[y0:y1, x0:x1]
        lo, hi = np.percentile(crop, 5), np.percentile(crop, 99.5)
        ax.imshow(np.clip((crop - lo) / (hi - lo), 0, 1) ** 0.5, cmap="gray", extent=(x0, x1, y1, y0), aspect="auto")
        xs = np.array([x0, x1])
        ax.plot(xs, jy + r["slope"] * (xs - jx), "c-", lw=0.6, alpha=0.6)
        ax.plot(jx, jy, "c+", ms=10)
        for cx, cy in r["cands"]:
            ax.plot(cx, cy, "o", mfc="none", mec="red", ms=7, mew=0.6)
        for mx, my, _ in r["moons"]:
            ax.plot(mx, my, "o", mfc="none", mec="yellow", ms=13)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{f[:2]} Jul: {len(r['moons'])} moons", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("auto_detections.png", dpi=110)

    # sine fits
    cost, params, label, errs = fit_waves(t, y, frame, np.random.default_rng(SEED))
    print(f"\n{len(t)} detections, RMS residual {np.sqrt(cost / len(t)):.2f} Jupiter radii\n")
    print(f"{'moon':<16}{'period (d)':>12}{'amplitude (RJ)':>18}{'n pts':>7}")
    for k, ((name, _, _), p, e) in enumerate(zip(MOON_BOUNDS, params, errs)):
        print(f"{name:<16}{p[0]:>8.3f}±{e[0]:.3f}{p[1]:>11.2f}±{e[1]:.2f}{(label == k).sum():>7}")

    tt = np.linspace(t_all.min(), t_all.max(), 1000)
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = plt.axes([0.12, 0.12, 0.83, 0.83])
    ax.scatter(t[label < 0], y[label < 0], c="gray")
    for k, (p, c) in enumerate(zip(params, COLORS)):
        ax.scatter(t[label == k], y[label == k], c=c)
        ax.plot(tt, wave(p, tt), c=c)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Day")
    ax.set_ylabel("Jupiter Radii")
    fig.savefig("auto_moon_fit.png", dpi=150)

    # overlay: automatic fit (dashed) on your manual fit (solid)
    man_params, mt, my, mlabel = manual_waves()
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = plt.axes([0.12, 0.12, 0.83, 0.83])
    for k, c in enumerate(COLORS):
        ax.scatter(mt[mlabel == k], my[mlabel == k], c=c, marker="o", s=25, alpha=0.5)
        ax.scatter(t[label == k], y[label == k], facecolors="none", edgecolors=c, marker="D", s=40)
        ax.plot(tt, wave(man_params[k], tt), c=c, ls="-", lw=1.5)
        ax.plot(tt, wave(params[k], tt), c=c, ls="--", lw=1.5)
    ax.plot([], [], "k-o", ms=5, label="Manual")
    ax.plot([], [], "k--D", ms=6, mfc="none", label="Automatic")
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Day")
    ax.set_ylabel("Jupiter Radii")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig("auto_vs_manual_fit.png", dpi=150)
    print("Manual vs automatic (P d, A RJ):")
    for k, (m, a_) in enumerate(zip(man_params, params)):
        print(f"  Moon {k + 1}: manual P={m[0]:.3f} A={m[1]:.2f}   auto P={a_[0]:.3f} A={a_[1]:.2f}")

    # power law from the fitted periods / amplitudes
    period = np.array([p[0] for p in params])
    axis = np.array([p[1] for p in params])
    x, C, b, x_err, lx, ly = power_law(period, axis)
    mx_, mC, mb, _, mlx, mly = manual_power_law()
    print(f"\nAutomatic: P = {C:.4f} * a^{x:.3f}   (log10 C = {b:.3f})")
    print(f"Manual:    P = {mC:.4f} * a^{mx_:.3f}   (log10 C = {mb:.3f})")
    x3 = power_law(period[1:], axis[1:])[0]
    print(f"Moons 2-4 only (no inner-moon alias prior): exponent {x3:.3f}")
    print("Kepler's 3rd law predicts an exponent of 1.5")

    fig, ax = plt.subplots(figsize=(7, 5))
    allx = np.concatenate([lx, mlx])
    span = allx.max() - allx.min()
    xl = np.array([min(0.0, allx.min()) - 0.15 * span, allx.max() + 0.15 * span])
    ax.plot(xl, mb + mx_ * xl, "-", color="#0b3d91", label=f"Manual fit (slope {mx_:.3f})", zorder=1)
    ax.plot(xl, b + x * xl, "--", color="#FC3D21", label=f"Automatic fit (slope {x:.3f})", zorder=1)
    ax.scatter(mlx, mly, color="#0b3d91", label="Manual observations", zorder=2)
    ax.scatter(lx, ly, facecolors="none", edgecolors="#FC3D21", marker="D", s=60,
               label="Automatic observations", zorder=3)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("log(a) [Jupiter radii]")
    ax.set_ylabel("log(P) [days]")
    ax.set_title("Orbital Period vs. Semi-Major Axis (log-log)")
    ax.set_xlim(*xl)
    ax.legend(loc="upper left")
    fig.savefig("auto_power_law_fit.png", dpi=200)

    print("Saved auto_detections.png, auto_moon_fit.png, auto_vs_manual_fit.png, auto_power_law_fit.png, auto_positions.csv")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
