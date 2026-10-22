# fully automatic Lab 2: images -> moon positions -> sine fits -> power law
# run from this folder: python auto_pipeline.py [--no-show]

import csv
import glob
import sys
from typing import cast

import matplotlib.pyplot as plot
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment

posCsv = "pos.csv"
obsCsv = "observations.csv"
imageGlob = "*-SHIFT.JPG"

pixelSize = 4.289e-3 # mm
au = 1.496e8 # km
jupiterRadius = 7.1492e4 # km

jupiterSaturation = 250 # brightness (0-255) of Jupiter's saturated disc
boxHalfWidth = 700 # px either side of Jupiter searched for moons
boxHalfHeight = 250 # px above/below Jupiter
dogSmall, dogLarge = 1.5, 5.0 # difference-of-Gaussians scales, px
peakSize = 9 # px, closer peaks are merged
minSnr = 9.0 # DoG peak height over the frame's robust noise
minAxisDx = 14 # px, anything closer to Jupiter is a glint
maxElongation = 32.0 # Jupiter radii, Callisto's orbit is ~26
maxTilt = 0.35 # max |dy/dx| of the moon axis, ~19 degrees
axisTol = 12.0 # px a moon may sit off the axis
axisTolFrac = 0.03 # plus this fraction of its distance from Jupiter

aliasCostRatio = 3.0 # inner moon periods this close to the best fit count as aliases
outlierDistance = 3.0 # Jupiter radii from its wave before a point is a false detection

# (name, period bounds in days, amplitude bounds in Jupiter radii), innermost first
moonBounds = [
    ("Moon 1 (inner)", (1.0, 2.5), (3.0, 9.0)),
    ("Moon 2", (2.5, 5.0), (7.0, 13.0)),
    ("Moon 3", (5.0, 10.0), (11.0, 19.0)),
    ("Moon 4 (outer)", (10.0, 25.0), (20.0, 32.0)),
]

startCount = 400
maxIters = 30
seed = 0

colors = ["red", "green", "blue", "purple"]
nasaBlue, nasaRed = "#0B3D91", "#FC3D21"
brightYellow = "#FFFF00"

# centre and radius of the largest saturated blob
def findJupiter(pixels):
    blobs, blobCount = cast(tuple[np.ndarray, int], ndimage.label(pixels >= jupiterSaturation))
    blobSizes = ndimage.sum(np.ones_like(pixels), blobs, range(1, blobCount + 1))

    mask = blobs == (1 + int(np.argmax(blobSizes)))
    maskY, maskX = np.nonzero(mask)

    return maskX.mean(), maskY.mean(), np.sqrt(mask.sum() / np.pi)

# moons are (x, y, offset px along the tilted axis, negative = left of Jupiter)
def detect(path, maxOffsetPx):
    pixels = np.asarray(Image.open(path).convert("L"), dtype = float)
    jupiterX, jupiterY, _ = findJupiter(pixels)

    boxX, boxY = int(jupiterX) - boxHalfWidth, int(jupiterY) - boxHalfHeight
    box = pixels[max(boxY, 0):boxY + 2 * boxHalfHeight, max(boxX, 0):boxX + 2 * boxHalfWidth]
    boxX, boxY = max(boxX, 0), max(boxY, 0)

    # keeps moon-sized blobs, drops sky gradients and most of Jupiter's halo
    dog = ndimage.gaussian_filter(box, dogSmall) - ndimage.gaussian_filter(box, dogLarge)
    noise = 1.4826 * np.median(np.abs(dog - np.median(dog)))
    snr = dog / noise

    isPeak = (dog == ndimage.maximum_filter(dog, size = peakSize)) & (snr > minSnr)

    candidates = [] # (dx, dy, snr) relative to Jupiter

    for peakY, peakX in zip(*np.nonzero(isPeak)):
        rowSlice, colSlice = slice(max(peakY - 3, 0), peakY + 4), slice(max(peakX - 3, 0), peakX + 4)
        weights = dog[rowSlice, colSlice].clip(0)
        gridY, gridX = np.mgrid[rowSlice, colSlice]

        centroidX = (weights * gridX).sum() / weights.sum() + boxX - jupiterX
        centroidY = (weights * gridY).sum() / weights.sum() + boxY - jupiterY

        candidates.append((centroidX, centroidY, snr[peakY, peakX]))

    candidates = [candidate for candidate in candidates if minAxisDx <= abs(candidate[0]) <= maxOffsetPx]

    # the line through Jupiter with the brightest peaks rejects stray stars and fixes camera tilt
    def members(slope):
        return [candidate for candidate in candidates if abs(candidate[1] - slope * candidate[0]) <= axisTol + axisTolFrac * abs(candidate[0])]

    def brightestMembers(slope):
        return sorted(members(slope), key = lambda candidate: -candidate[2])[:4]

    slopes = np.linspace(-maxTilt, maxTilt, 141)
    scores = [sum(np.log(candidate[2]) for candidate in brightestMembers(slope)) for slope in slopes]

    slope = slopes[int(np.argmax(scores))]
    moonCandidates = brightestMembers(slope)

    if len(moonCandidates) >= 2:
        offsetX = np.array([candidate[0] for candidate in moonCandidates])
        offsetY = np.array([candidate[1] for candidate in moonCandidates])

        slope = (offsetX * offsetY).sum() / (offsetX * offsetX).sum()

    axisDirection = np.array([1.0, slope]) / np.hypot(1.0, slope)
    moons = sorted((candidate[0] * axisDirection[0] + candidate[1] * axisDirection[1], jupiterX + candidate[0], jupiterY + candidate[1]) for candidate in moonCandidates)

    return {"jupiter": (jupiterX, jupiterY), "slope": slope, "moons": [(moonX, moonY, offset) for offset, moonX, moonY in moons],
            "candidates": [(jupiterX + candidate[0], jupiterY + candidate[1]) for candidate in candidates]}

# time, focal length and distance (AU) per image, in image order
def loadMetadata(path):
    with open(path, newline = "") as data:
        rows = list(csv.DictReader(data))

    times = np.array([float(row["YYYYDDD.DDD"]) for row in rows])
    focalLengths = np.array([float(row["FL"]) for row in rows])
    distances = np.array([float(row["Distance"]) for row in rows])

    return times - times[0], focalLengths, distances

def measureAll():
    fileNames = sorted(glob.glob(imageGlob), key = lambda fileName: (fileName[2:4], fileName[0:2])) # by month, day
    imageTimes, focalLengths, distances = loadMetadata(posCsv)

    if len(fileNames) != len(imageTimes):
        sys.exit(f"{len(fileNames)} images but {len(imageTimes)} rows in {posCsv}")

    scales = pixelSize / focalLengths * distances * au / jupiterRadius # Jupiter radii/px
    results = []

    for fileName, scale in zip(fileNames, scales):
        result = detect(fileName, maxElongation / scale)
        results.append(result)

        print(f"{fileName}: Jupiter at ({result['jupiter'][0]:.0f},{result['jupiter'][1]:.0f}), "
              f"axis tilt {np.degrees(np.arctan(result['slope'])):+.1f} deg, {len(result['moons'])} moons found")

    times, offsets, frames = [], [], []

    for imageIndex, result in enumerate(results):
        for _, _, offsetPx in result["moons"]:
            times.append(imageTimes[imageIndex])
            offsets.append(offsetPx * scales[imageIndex])
            frames.append(imageIndex)

    return fileNames, results, imageTimes, np.array(times), np.array(offsets), np.array(frames)

def wave(moon, times):
    period, amplitude, phase = moon
    return amplitude * np.sin(2 * np.pi * times / period + phase)

def assign(params, times, offsets, frames):
    labels = np.full(len(times), -1)

    for frame in np.unique(frames):
        indices = np.where(frames == frame)[0]
        costMatrix = np.array([[(offsets[i] - wave(moon, times[i])) ** 2 for moon in params] for i in indices])

        pointIndices, moonIndices = linear_sum_assignment(costMatrix)
        labels[indices[pointIndices]] = moonIndices

    for i in np.nonzero(labels >= 0)[0]: # reject false detections
        if abs(offsets[i] - wave(params[labels[i]], times[i])) > outlierDistance:
            labels[i] = -1

    return labels

def refit(initial, times, offsets, bounds):
    periodBounds, amplitudeBounds = bounds
    lower = np.array([periodBounds[0], amplitudeBounds[0], -np.inf])
    upper = np.array([periodBounds[1], amplitudeBounds[1], np.inf])

    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)

    return least_squares(lambda moon: wave(moon, times) - offsets, initial, bounds = (lower, upper))

# rejected points cost the full outlier penalty
def totalCost(params, times, offsets, labels):
    cost = outlierDistance ** 2 * np.sum(labels < 0)

    for moonIndex, moon in enumerate(params):
        selected = labels == moonIndex
        cost += np.sum((offsets[selected] - wave(moon, times[selected])) ** 2)

    return cost

# daily sampling aliases the inner moon, so grid search its period over the points moons 2-4 leave
def refineInner(params, labels, times, offsets, frames, cost):
    unclaimed = (labels < 0) | (labels == 0)
    unclaimedTimes, unclaimedOffsets = times[unclaimed], offsets[unclaimed]

    (_, (minPeriod, maxPeriod), (minAmplitude, maxAmplitude)) = moonBounds[0]
    trialPeriods = np.arange(minPeriod, maxPeriod, 0.002)
    costs, solutions = [], []

    for period in trialPeriods:
        design = np.column_stack([np.sin(2 * np.pi * unclaimedTimes / period), np.cos(2 * np.pi * unclaimedTimes / period)])
        coefficients = np.linalg.lstsq(design, unclaimedOffsets, rcond = None)[0]
        residuals = unclaimedOffsets - design @ coefficients

        for _ in range(4): # robust re-weighting
            weights = 1 / (1 + residuals ** 2)
            coefficients = np.linalg.lstsq(design * weights[:, None], unclaimedOffsets * weights, rcond = None)[0]

            residuals = unclaimedOffsets - design @ coefficients

        amplitude = np.hypot(*coefficients)

        costs.append(np.sum(np.minimum(residuals ** 2, outlierDistance ** 2)) if minAmplitude <= amplitude <= maxAmplitude else np.inf)
        solutions.append(np.array([period, amplitude, np.arctan2(coefficients[1], coefficients[0])]))

    costs = np.array(costs)
    minima = [i for i in range(1, len(costs) - 1) if costs[i] < costs[i - 1] and costs[i] <= costs[i + 1]]

    if not minima:
        return cost, params, labels

    best = min(minima, key = lambda i: costs[i])

    # among near-best aliases, pick the one closest to a power law through moons 2-4
    exponent, intercept = np.polyfit(np.log([moon[1] for moon in params[1:]]), np.log([moon[0] for moon in params[1:]]), 1)
    nearBest = [i for i in minima if costs[i] <= aliasCostRatio * costs[best]]
    pick = min(nearBest, key = lambda i: abs(np.log(solutions[i][0]) - (intercept + exponent * np.log(solutions[i][1]))))

    if pick != best:
        print(f"Inner moon: data alone favour the alias P = {solutions[best][0]:.3f} d; "
              f"using P = {solutions[pick][0]:.3f} d, the minimum closest to the outer-moon power law.")

    newParams = list(params)
    newParams[0] = solutions[pick]

    for _ in range(maxIters):
        newLabels = assign(newParams, times, offsets, frames)

        for moonIndex, (_, periodBounds, amplitudeBounds) in enumerate(moonBounds):
            selected = newLabels == moonIndex

            if selected.sum() >= 4:
                newParams[moonIndex] = refit(newParams[moonIndex], times[selected], offsets[selected], (periodBounds, amplitudeBounds)).x

        if np.array_equal(newLabels, assign(newParams, times, offsets, frames)):
            break

    newLabels = assign(newParams, times, offsets, frames)
    newCost = totalCost(newParams, times, offsets, newLabels)

    return (newCost, newParams, newLabels) if newCost < cost else (cost, params, labels)

def fitWaves(times, offsets, frames, random):
    best = (np.inf, None, None)

    for _ in range(startCount):
        params = [
            np.array([random.uniform(*periodBounds), random.uniform(*amplitudeBounds), random.uniform(0, 2 * np.pi)])
            for _, periodBounds, amplitudeBounds in moonBounds
        ]
        labels = assign(params, times, offsets, frames)

        for _ in range(maxIters):
            newParams = []

            for moonIndex, (_, periodBounds, amplitudeBounds) in enumerate(moonBounds):
                selected = labels == moonIndex

                if selected.sum() < 4:
                    newParams.append(params[moonIndex])
                else:
                    newParams.append(refit(params[moonIndex], times[selected], offsets[selected], (periodBounds, amplitudeBounds)).x)

            params = newParams
            newLabels = assign(params, times, offsets, frames)

            if np.array_equal(newLabels, labels):
                break

            labels = newLabels

        cost = totalCost(params, times, offsets, labels)

        if cost < best[0]:
            best = (cost, params, labels)

    cost, params, labels = best
    cost, params, labels = refineInner(params, labels, times, offsets, frames, cost)

    uncertainties = [] # 1-sigma on period and amplitude from the jacobian

    for moonIndex, (_, periodBounds, amplitudeBounds) in enumerate(moonBounds):
        selected = labels == moonIndex
        result = refit(params[moonIndex], times[selected], offsets[selected], (periodBounds, amplitudeBounds))

        degreesOfFreedom = max(selected.sum() - 3, 1)
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * (2 * result.cost / degreesOfFreedom)

        uncertainties.append(np.sqrt(np.diag(covariance)))

    return cost, params, labels, uncertainties

# log10 P = log10 C + x log10 a
def powerLaw(periods, semiMajorAxes):
    logAxes, logPeriods = np.log10(semiMajorAxes), np.log10(periods)

    if len(logAxes) > 3: # polyfit needs more than 3 points for a covariance
        (exponent, intercept), covariance = np.polyfit(logAxes, logPeriods, 1, cov = True)
        exponentError = np.sqrt(covariance[0, 0])
    else:
        exponent, intercept = np.polyfit(logAxes, logPeriods, 1)
        exponentError = np.nan

    return exponent, 10 ** intercept, intercept, exponentError, logAxes, logPeriods

def manualPowerLaw():
    observations = np.genfromtxt(obsCsv, delimiter = ",", skip_header = 1)
    return powerLaw(observations[:, 1], observations[:, 2])

# hand-fitted period and amplitude from observations.csv, phases weren't saved so only those are solved
def manualWaves():
    from fit_moons import load_points

    times, offsets, frames, _ = load_points(posCsv)
    observations = np.genfromtxt(obsCsv, delimiter = ",", skip_header = 1)

    periodAmplitudes = sorted((row[1], row[2]) for row in observations) # shortest period first
    trialPhases = np.linspace(0, 2 * np.pi, 720, endpoint = False)

    def phaseCost(period, amplitude, phase): # closest point to the wave in each frame
        cost = 0

        for frame in np.unique(frames):
            selected = frames == frame
            cost += min(np.min((offsets[selected] - wave((period, amplitude, phase), times[selected])) ** 2), outlierDistance ** 2)

        return cost

    params = []

    for period, amplitude in periodAmplitudes:
        bestPhase = trialPhases[np.argmin([phaseCost(period, amplitude, phase) for phase in trialPhases])]
        params.append(np.array([period, amplitude, bestPhase]))

    labels = assign(params, times, offsets, frames)

    for _ in range(maxIters):
        newParams = []

        for moonIndex, (period, amplitude, _) in enumerate(params):
            selected = labels == moonIndex
            costs = [np.sum((offsets[selected] - wave((period, amplitude, phase), times[selected])) ** 2) for phase in trialPhases]

            newParams.append(np.array([period, amplitude, trialPhases[np.argmin(costs)]]))

        if all(np.allclose(new, old) for new, old in zip(newParams, params)):
            break

        params = newParams
        labels = assign(params, times, offsets, frames)

    return params, times, offsets, labels

# crop with the tilted axis, candidate peaks (yellow) and accepted moons (red)
def drawDetections(ax, fileName, result, title, titleSize):
    pixels = np.asarray(Image.open(fileName).convert("L"), dtype = float)
    jupiterX, jupiterY = result["jupiter"]

    left, right = int(jupiterX) - boxHalfWidth, int(jupiterX) + boxHalfWidth
    top, bottom = int(jupiterY) - boxHalfHeight // 2, int(jupiterY) + boxHalfHeight // 2
    crop = pixels[top:bottom, left:right]

    darkLevel, brightLevel = np.percentile(crop, 5), np.percentile(crop, 99.5)
    stretched = np.clip((crop - darkLevel) / (brightLevel - darkLevel), 0, 1) ** 0.5

    ax.imshow(stretched, cmap = "gray", extent = (left, right, bottom, top), aspect = "auto")

    axisX = np.array([left, right])
    ax.plot(axisX, jupiterY + result["slope"] * (axisX - jupiterX), "-", color = nasaBlue, lw = 1.2)
    ax.plot(jupiterX, jupiterY, "+", color = nasaBlue, ms = 10)

    for candidateX, candidateY in result["candidates"]:
        ax.plot(candidateX, candidateY, "o", mfc = "none", mec = brightYellow, ms = 7, mew = 0.6)

    for moonX, moonY, _ in result["moons"]:
        ax.plot(moonX, moonY, "o", mfc = "none", mec = nasaRed, ms = 13)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_title(title, fontsize = titleSize)
    ax.axis("off")

def main():
    show = "--no-show" not in sys.argv
    fileNames, results, imageTimes, times, offsets, frames = measureAll()

    with open("auto_positions.csv", "w", newline = "") as data:
        writer = csv.writer(data)
        writer.writerow(["image", "day", "offset_RJ"])

        for frame, time, offset in zip(frames, times, offsets):
            writer.writerow([fileNames[frame], f"{time:.4f}", f"{offset:.3f}"])

    fig, axes = plot.subplots(4, 4, figsize = (22, 9))

    for ax, fileName, result in zip(axes.flat, fileNames, results):
        drawDetections(ax, fileName, result, f"{fileName[:2]} Jul: {len(result['moons'])} moons", 9)

    fig.tight_layout()
    fig.savefig("auto_detections.png", dpi = 110)

    jul28Index = fileNames.index("28072019-1-SHIFT.JPG")
    fig, ax = plot.subplots(figsize = (10, 4.5))
    drawDetections(ax, fileNames[jul28Index], results[jul28Index], "July 28, 2019", 14)

    fig.tight_layout()
    fig.savefig("auto_detections_jul28.png", dpi = 150)

    cost, params, labels, uncertainties = fitWaves(times, offsets, frames, np.random.default_rng(seed))

    print(f"\n{len(times)} detections, RMS residual {np.sqrt(cost / len(times)):.2f} Jupiter radii\n")
    print(f"{'moon':<16}{'period (d)':>12}{'amplitude (RJ)':>18}{'n pts':>7}")

    for moonIndex, ((name, _, _), moon, uncertainty) in enumerate(zip(moonBounds, params, uncertainties)):
        print(f"{name:<16}{moon[0]:>8.3f}±{uncertainty[0]:.3f}{moon[1]:>11.2f}±{uncertainty[1]:.2f}{(labels == moonIndex).sum():>7}")

    smoothTimes = np.linspace(imageTimes.min(), imageTimes.max(), 1000)

    fig = plot.figure(figsize = (8, 5), dpi = 100)
    ax = plot.axes((0.12, 0.12, 0.83, 0.83))
    ax.scatter(times[labels < 0], offsets[labels < 0], c = "gray")

    for moonIndex, (moon, color) in enumerate(zip(params, colors)):
        ax.scatter(times[labels == moonIndex], offsets[labels == moonIndex], c = color)
        ax.plot(smoothTimes, wave(moon, smoothTimes), c = color)

    ax.set_ylim(-30, 30)
    ax.set_xlabel("Day")
    ax.set_ylabel("Jupiter Radii")
    fig.savefig("auto_moon_fit.png", dpi = 150)

    # automatic fit (dashed) over the manual fit (solid)
    manualParams, manualTimes, manualOffsets, manualLabels = manualWaves()

    fig = plot.figure(figsize = (8, 5), dpi = 100)
    ax = plot.axes((0.12, 0.12, 0.83, 0.83))

    for moonIndex, color in enumerate(colors):
        ax.scatter(manualTimes[manualLabels == moonIndex], manualOffsets[manualLabels == moonIndex], c = color, marker = "o", s = 25, alpha = 0.5)
        ax.scatter(times[labels == moonIndex], offsets[labels == moonIndex], facecolors = "none", edgecolors = color, marker = "D", s = 40)

        ax.plot(smoothTimes, wave(manualParams[moonIndex], smoothTimes), c = color, ls = "-", lw = 1.5)
        ax.plot(smoothTimes, wave(params[moonIndex], smoothTimes), c = color, ls = "--", lw = 1.5)

    ax.plot([], [], "k-o", ms = 5, label = "Manual")
    ax.plot([], [], "k--D", ms = 6, mfc = "none", label = "Automatic")

    ax.set_ylim(-30, 30)
    ax.set_xlabel("Day")
    ax.set_ylabel("Jupiter Radii")
    ax.legend(loc = "lower right", fontsize = 8)
    fig.savefig("auto_vs_manual_fit.png", dpi = 150)

    print("Manual vs automatic (P d, A RJ):")

    for moonIndex, (manual, auto) in enumerate(zip(manualParams, params)):
        print(f"  Moon {moonIndex + 1}: manual P={manual[0]:.3f} A={manual[1]:.2f}   auto P={auto[0]:.3f} A={auto[1]:.2f}")

    periods = np.array([moon[0] for moon in params])
    semiMajorAxes = np.array([moon[1] for moon in params])

    with open("auto_observations.csv", "w", newline = "") as data: # same layout as observations.csv, outermost first
        writer = csv.writer(data)
        writer.writerow(["moon", "orbital period (days)", "semi-major axis (Jupiter radii)"])

        for moonNumber, moonIndex in enumerate(np.argsort(-periods), start = 1):
            writer.writerow([moonNumber, f"{periods[moonIndex]:.3f}", f"{semiMajorAxes[moonIndex]:.3f}"])

    exponent, constant, intercept, exponentError, logAxes, logPeriods = powerLaw(periods, semiMajorAxes)
    manualExponent, manualConstant, manualIntercept, _, manualLogAxes, manualLogPeriods = manualPowerLaw()
    outerExponent = powerLaw(periods[1:], semiMajorAxes[1:])[0]

    print(f"\nAutomatic: P = {constant:.4f} * a^{exponent:.3f}   (log10 C = {intercept:.3f})")
    print(f"Manual:    P = {manualConstant:.4f} * a^{manualExponent:.3f}   (log10 C = {manualIntercept:.3f})")
    print(f"Moons 2-4 only (no inner-moon alias prior): exponent {outerExponent:.3f}")
    print("Kepler's 3rd law predicts an exponent of 1.5")

    allLogAxes = np.concatenate([logAxes, manualLogAxes])
    span = allLogAxes.max() - allLogAxes.min()
    xLimits = np.array([min(0.0, allLogAxes.min()) - 0.15 * span, allLogAxes.max() + 0.15 * span])

    fig, ax = plot.subplots(figsize = (7, 5))

    ax.plot(xLimits, manualIntercept + manualExponent * xLimits, "-", color = nasaBlue, label = f"Manual fit (slope {manualExponent:.3f})", zorder = 1)
    ax.plot(xLimits, intercept + exponent * xLimits, "--", color = nasaRed, label = f"Automatic fit (slope {exponent:.3f})", zorder = 1)

    ax.scatter(manualLogAxes, manualLogPeriods, color = nasaBlue, label = "Manual observations", zorder = 2)
    ax.scatter(logAxes, logPeriods, facecolors = "none", edgecolors = nasaRed, marker = "D", s = 60, label = "Automatic observations", zorder = 3)

    ax.axvline(0, color = "gray", lw = 0.8)
    ax.axhline(0, color = "gray", lw = 0.8)

    ax.set_xlabel("log(a) [Jupiter radii]")
    ax.set_ylabel("log(P) [days]")
    ax.set_title("Orbital Period vs. Semi-Major Axis (log-log)")
    ax.set_xlim(*xLimits)
    ax.legend(loc = "upper left")
    fig.savefig("auto_power_law_fit.png", dpi = 200)

    print("Saved auto_detections.png, auto_detections_jul28.png, auto_moon_fit.png, auto_vs_manual_fit.png, auto_power_law_fit.png, auto_positions.csv, auto_observations.csv")

    if show:
        plot.show()

if __name__ == "__main__":
    main()
