import csv
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from craterData import gravity, loadImpacts, writeCsv

sys.stdout.reconfigure(encoding = "utf-8") # so ± prints on Windows consoles

# Power laws: D = C * m^alpha, D = C * v^beta
# Linearize: log(D) = log(C) + alpha * log(m), and the same for v, E, p

fitColor = "#0b3d91"
dataColor = "#FC3D21"
densityColor = "#1B998B"
velocityColor = "#F2A900"
densityText = "#0f6b61" # darker teal for text

lineExtension = 0.15 # fraction of the data span added to each end of a fit line


class LineFit:
    """Ordinary least-squares line y = slope * x + intercept."""

    def __init__(self, x, y):
        result = stats.linregress(x, y)
        self.n = len(x)
        self.slope = result.slope
        self.slopeError = result.stderr
        self.intercept = result.intercept
        self.interceptError = result.intercept_stderr
        self.rSquared = result.rvalue ** 2

    def predict(self, x):
        return self.slope * x + self.intercept

    def describe(self):
        return (f"n = {self.n:2d}  slope = {self.slope:.4f} ± {self.slopeError:.4f}  "
                f"intercept = {self.intercept:.4f} ± {self.interceptError:.4f}  R^2 = {self.rSquared:.4f}")


def fitLine(x, y):
    return LineFit(x, y)


def ratioError(ratio, numerator, numeratorError, denominator, denominatorError):
    return abs(ratio) * math.sqrt((numeratorError / numerator) ** 2 + (denominatorError / denominator) ** 2)


# ---- data ----

data = loadImpacts()
writeCsv(data) # cratering.csv: the raw measurements pulled from Cratering.xlsx

impact = data["impact"]
impactSet = data["set"]
material = data["material"]
sandDepth = data["sandDepth"] # cm
dropHeight = data["dropHeight"] # cm
fallHeight = data["fallHeight"] # cm
ballDiameter = data["ballDiameter"] # cm
mass = data["mass"] # g
craterDiameter = data["craterDiameter"] # mm

# ---- derived quantities ----

velocity = np.sqrt(2 * gravity * fallHeight) # cm/s
ballVolume = 4 / 3 * np.pi * (ballDiameter / 2) ** 3 # cm^3
density = mass / ballVolume # g/cm^3

massKg = mass / 1000.0 # kg
velocitySi = velocity / 100.0 # m/s
energy = 0.5 * massKg * velocitySi ** 2 # J
momentum = massKg * velocitySi # kg m/s

logMass = np.log10(mass) # g
logVelocity = np.log10(velocitySi) # m/s
logDiameter = np.log10(craterDiameter) # mm
logEnergy = np.log10(energy) # J
logMomentum = np.log10(momentum) # kg m/s


def impacts(*numbers):
    return np.isin(impact, numbers)


massSet = impactSet == "mass"
densitySet = impactSet == "density"
velocitySet = impactSet == "velocity"
allImpacts = np.ones(len(impact), dtype = bool)

# ---- fits ----

massFit = fitLine(logMass[massSet], logDiameter[massSet])
alpha = massFit.slope
alphaError = massFit.slopeError

velocityFit = fitLine(logVelocity[velocitySet], logDiameter[velocitySet])
beta = velocityFit.slope
betaError = velocityFit.slopeError

densityFit = fitLine(logMass[densitySet], logDiameter[densitySet])
massAndDensity = massSet | densitySet
overallFit = fitLine(logMass[massAndDensity], logDiameter[massAndDensity]) # impacts 1-10, plotted only

energyFit = fitLine(logEnergy, logDiameter)
momentumFit = fitLine(logMomentum, logDiameter)

# sensitivity checks
noImpact11 = velocitySet & ~impacts(11)
velocityFitNo11 = fitLine(logVelocity[noImpact11], logDiameter[noImpact11])

leaveOneOut = []
for leftOut in impact[velocitySet]:
    kept = velocitySet & ~impacts(leftOut)
    leaveOneOut.append((leftOut, fitLine(logVelocity[kept], logDiameter[kept]).slope))

withImpact7 = velocitySet | impacts(7)
velocityFitWith7 = fitLine(logVelocity[withImpact7], logDiameter[withImpact7])

# multivariate: log D = c + alpha log m + beta log v, all impacts
design = np.column_stack([np.ones(len(impact)), logMass, logVelocity])
coefficients, _, _, _ = np.linalg.lstsq(design, logDiameter, rcond = None)
multiResiduals = logDiameter - design @ coefficients
residualVariance = np.sum(multiResiduals ** 2) / (len(impact) - 3)
covariance = residualVariance * np.linalg.inv(design.T @ design)
coefficientErrors = np.sqrt(np.diag(covariance))

# ---- comparisons ----

gamma = densityFit.slope - alpha
gammaError = math.sqrt(densityFit.slopeError ** 2 + alphaError ** 2)

predictedDensityDiameter = 10 ** massFit.predict(logMass[densitySet]) # mm
densityResidualDex = logDiameter[densitySet] - np.log10(predictedDensityDiameter)
densityDeviation = (craterDiameter[densitySet] / predictedDensityDiameter - 1) * 100 # %

massResidualDex = logDiameter[massSet] - massFit.predict(logMass[massSet])
massResidualRms = math.sqrt(np.mean(massResidualDex ** 2))

ratio = beta / alpha
ratioSigma = ratioError(ratio, beta, betaError, alpha, alphaError)

betaMinusAlpha = beta - alpha
betaMinusAlphaError = math.sqrt(alphaError ** 2 + betaError ** 2)
betaMinusTwoAlpha = beta - 2 * alpha
betaMinusTwoAlphaError = math.sqrt(4 * alphaError ** 2 + betaError ** 2)

repeated = impacts(4, 7, 11) # 1.27 cm steel from 91.1 cm
repeatedDiameters = craterDiameter[repeated]
repeatedMean = np.mean(repeatedDiameters)
repeatedStd = np.std(repeatedDiameters, ddof = 1)

massRange = mass.max() / mass.min()
velocityRange = velocity[velocitySet].max() / velocity[velocitySet].min()

# ---- summary ----

def section(title):
    print()
    print(title)
    print("-" * len(title))


print(f"Lab 3 cratering analysis  (g = {gravity:g} cm/s^2, {len(impact)} impacts)")

section("Derived quantities")
print("impact  set       material  fall (cm)  v (cm/s)  rho (g/cm^3)  E (J)      p (kg m/s)  D (mm)")
for index in range(len(impact)):
    print(f"{impact[index]:6d}  {impactSet[index]:8s}  {material[index]:8s}  {fallHeight[index]:9.1f}  "
          f"{velocity[index]:8.1f}  {density[index]:12.2f}  {energy[index]:<9.4g}  {momentum[index]:<10.4g}  {craterDiameter[index]:6g}")

section("Fits (log10 D [mm] against log10 x)")
print(f"Mass, 1-6 (alpha):       {massFit.describe()}")
print(f"Velocity, 11-17 (beta):  {velocityFit.describe()}")
print(f"Velocity, 12-17:         {velocityFitNo11.describe()}")
print(f"Density set, 7-10:       {densityFit.describe()}")
print(f"Mass + density, 1-10:  {overallFit.describe()}")
print(f"Energy, all 17:          {energyFit.describe()}")
print(f"Momentum, all 17:        {momentumFit.describe()}")

section("Density beyond mass")
print(f"Density-set slope = alpha + gamma, so gamma = {gamma:.4f} ± {gammaError:.4f}")
print("material  m (g)   rho (g/cm^3)  D (mm)  predicted D (mm)  deviation  residual (dex)")
for index, densityIndex in enumerate(np.flatnonzero(densitySet)):
    print(f"{material[densityIndex]:8s}  {mass[densityIndex]:5.2f}  {density[densityIndex]:12.2f}  "
          f"{craterDiameter[densityIndex]:6g}  {predictedDensityDiameter[index]:16.1f}  "
          f"{densityDeviation[index]:+8.1f}%  {densityResidualDex[index]:+.3f}")

section("Mass-set residuals")
for number, residual in zip(impact[massSet], massResidualDex):
    print(f"impact {number:2d}: {residual:+.3f} dex")
print(f"RMS: {massResidualRms:.3f} dex")

section("Momentum vs. energy scaling")
print(f"beta / alpha = {ratio:.4f} ± {ratioSigma:.4f}")
print(f"  (r - 1) / sigma_r = {(ratio - 1) / ratioSigma:+.2f}   (momentum predicts 1)")
print(f"  (r - 2) / sigma_r = {(ratio - 2) / ratioSigma:+.2f}   (kinetic energy predicts 2)")
print(f"beta - alpha  = {betaMinusAlpha:.4f} ± {betaMinusAlphaError:.4f}")
print(f"beta - 2alpha = {betaMinusTwoAlpha:.4f} ± {betaMinusTwoAlphaError:.4f}")

section("Comparison with literature D ~ E^(1/4)")
print(f"(alpha - 0.25) / sigma_alpha = {(alpha - 0.25) / alphaError:+.2f}")
print(f"(beta - 0.5) / sigma_beta    = {(beta - 0.5) / betaError:+.2f}")
print(f"energy slope - 1/4           = {(energyFit.slope - 0.25) / energyFit.slopeError:+.2f} sigma")
print(f"energy slope - 1/3           = {(energyFit.slope - 1 / 3) / energyFit.slopeError:+.2f} sigma")

section("Sensitivity of beta")
print(f"Without impact 11:  beta = {velocityFitNo11.slope:.4f} ± {velocityFitNo11.slopeError:.4f}")
for leftOut, slope in leaveOneOut:
    print(f"Leave out {leftOut:2d}:     beta = {slope:.4f}")
print(f"With impact 7:      beta = {velocityFitWith7.slope:.4f} ± {velocityFitWith7.slopeError:.4f}")

section("Multivariate fit, all 17: log D = c + alpha log m + beta log v")
print(f"c     = {coefficients[0]:.4f} ± {coefficientErrors[0]:.4f}")
print(f"alpha = {coefficients[1]:.4f} ± {coefficientErrors[1]:.4f}")
print(f"beta  = {coefficients[2]:.4f} ± {coefficientErrors[2]:.4f}")

section("Repeated condition (1.27 cm steel from 91.1 cm: impacts 4, 7, 11)")
print(f"D = {', '.join(f'{value:g}' for value in repeatedDiameters)} mm")
print(f"mean = {repeatedMean:.1f} mm, std = {repeatedStd:.2f} mm ({repeatedStd / repeatedMean * 100:.1f}%)")

section("Ranges")
print(f"Mass, all impacts:     {massRange:.0f}x ({math.log10(massRange):.2f} dex)")
print(f"Velocity, 11-17:       {velocityRange:.2f}x ({math.log10(velocityRange):.2f} dex)")

# ---- tables ----

with open("derived.csv", "w", newline = "", encoding = "utf-8") as output:
    writer = csv.writer(output)
    writer.writerow([
        "impact", "set", "material",
        "sand depth (cm)", "drop height (cm)", "fall height (cm)",
        "ball diameter (cm)", "mass (g)", "density (g/cm^3)",
        "velocity (cm/s)", "energy (J)", "momentum (kg m/s)",
        "crater diameter (mm)",
        "log10 mass (g)", "log10 velocity (m/s)", "log10 diameter (mm)",
        "predicted diameter (mm)", "deviation from mass fit (%)",
    ])

    densityPosition = {densityIndex: index for index, densityIndex in enumerate(np.flatnonzero(densitySet))}

    for index in range(len(impact)):
        predicted = ""
        deviation = ""
        if index in densityPosition:
            predicted = f"{predictedDensityDiameter[densityPosition[index]]:.1f}"
            deviation = f"{densityDeviation[densityPosition[index]]:.1f}"

        writer.writerow([
            impact[index], impactSet[index], material[index],
            f"{sandDepth[index]:g}", f"{dropHeight[index]:g}", f"{fallHeight[index]:.1f}",
            f"{ballDiameter[index]:g}", f"{mass[index]:g}", f"{density[index]:.2f}",
            f"{velocity[index]:.1f}", f"{energy[index]:.4g}", f"{momentum[index]:.4g}",
            f"{craterDiameter[index]:g}",
            f"{logMass[index]:.4f}", f"{logVelocity[index]:.4f}", f"{logDiameter[index]:.4f}",
            predicted, deviation,
        ])

fitRows = [
    ("log D vs log m (alpha)", "1-6", massFit),
    ("log D vs log v (beta)", "11-17", velocityFit),
    ("log D vs log v", "12-17", velocityFitNo11),
    ("log D vs log m, density set", "7-10", densityFit),
    ("log D vs log E", "1-17", energyFit),
    ("log D vs log p", "1-17", momentumFit),
]

with open("fit_results.csv", "w", newline = "", encoding = "utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["fit", "impacts", "n", "slope", "slope std error", "intercept", "intercept std error", "R^2"])
    for name, impactRange, fit in fitRows:
        writer.writerow([
            name, impactRange, fit.n,
            f"{fit.slope:.4f}", f"{fit.slopeError:.4f}",
            f"{fit.intercept:.4f}", f"{fit.interceptError:.4f}",
            f"{fit.rSquared:.4f}",
        ])

# ---- figures ----

def drawFitLine(axis, fit, xData, color = fitColor, label = "Line of Best Fit", setLimits = True, margin = None):
    span = xData.max() - xData.min()
    if margin is None:
        margin = lineExtension * span # log units past each end of the data
    xLeft = xData.min() - margin
    xRight = xData.max() + margin
    xLine = np.array([xLeft, xRight])

    axis.plot(xLine, fit.predict(xLine), color = color, label = label, zorder = 1)
    if setLimits:
        axis.set_xlim(xLeft, xRight)


def annotateFit(axis, fit, symbol = None, y = 0.03, color = "black"):
    slopeName = f"Slope ({symbol})" if symbol else "Slope"
    text = f"R² = {fit.rSquared:.3f}\n{slopeName} = {fit.slope:.3f} ± {fit.slopeError:.3f}"
    axis.annotate(text, xy = (0.97, y), xycoords = "axes fraction", ha = "right", va = "bottom", color = color)


# 1. mass fit with the density set overlaid
figure, axis = plt.subplots()

densityMargin = lineExtension * (logMass[densitySet].max() - logMass[densitySet].min()) # same margin as the density line
drawFitLine(axis, overallFit, logMass[massAndDensity], color = fitColor, label = "Overall Fit")
drawFitLine(axis, massFit, logMass[massSet], color = dataColor, label = "Mass Fit", setLimits = False, margin = densityMargin)
drawFitLine(axis, densityFit, logMass[densitySet], color = densityColor, label = "Density Fit", setLimits = False)
axis.scatter(logMass[massSet], logDiameter[massSet], color = dataColor, label = "Mass Set", zorder = 2)
axis.scatter(logMass[densitySet], logDiameter[densitySet], color = densityColor, marker = "s", label = "Density Set", zorder = 3)

# label offsets (points) and alignment, chosen so neighbouring labels and the fit line don't collide
labelPlacement = {
    "Steel": ((8, -8), "left", "top"),
    "Brass": ((-6, 8), "right", "bottom"),
    "Teflon": ((8, -8), "left", "top"),
    "Nylon": ((-8, 6), "right", "bottom"),
}
labelBox = dict(boxstyle = "round,pad = 0.2", facecolor = "white", edgecolor = "none", alpha = 0.9)

for densityIndex in np.flatnonzero(densitySet):
    offset, horizontal, vertical = labelPlacement[material[densityIndex]]
    axis.annotate(material[densityIndex], xy = (logMass[densityIndex], logDiameter[densityIndex]), xytext = offset,
                  textcoords = "offset points", ha = horizontal, va = vertical, color = densityText, fontsize = 11,
                  bbox = labelBox, zorder = 5)

axis.set_xlabel("log(m) [g]")
axis.set_ylabel("log(D) [mm]")
axis.set_title("Crater Diameter vs. Impactor Mass (log-log)")
annotateFit(axis, overallFit, y = 0.31, color = fitColor)
annotateFit(axis, densityFit, "α + γ", y = 0.17, color = densityText)
annotateFit(axis, massFit, "α", y = 0.03, color = dataColor)
axis.legend(loc = "upper left")

figure.savefig("mass_density_fit.png", dpi = 300, bbox_inches = "tight")
plt.close(figure)

# 2. velocity fit
figure, axis = plt.subplots()

drawFitLine(axis, velocityFit, logVelocity[velocitySet])
axis.scatter(logVelocity[velocitySet], logDiameter[velocitySet], color = dataColor, label = "Velocity Set", zorder = 2)

axis.set_xlabel("log(v) [m/s]")
axis.set_ylabel("log(D) [mm]")
axis.set_title("Crater Diameter vs. Impact Velocity (log-log)")
annotateFit(axis, velocityFit, "β")
axis.legend(loc = "upper left")

figure.savefig("velocity_fit.png", dpi = 300, bbox_inches = "tight")
plt.close(figure)

# 3. energy and momentum, all impacts
setStyles = [
    (massSet, dataColor, "o", "Mass Set"),
    (densitySet, densityColor, "s", "Density Set"),
    (velocitySet, velocityColor, "^", "Velocity Set"),
]

figure, (energyAxis, momentumAxis) = plt.subplots(1, 2, figsize = (11, 4.8), sharey = True)

panels = [
    (energyAxis, logEnergy, energyFit, "log(E) [J]"),
    (momentumAxis, logMomentum, momentumFit, "log(p) [kg m/s]"),
]

for axis, xValues, fit, xLabel in panels:
    drawFitLine(axis, fit, xValues)
    for mask, color, marker, label in setStyles:
        axis.scatter(xValues[mask], logDiameter[mask], color = color, marker = marker, label = label, zorder = 2)
    axis.set_xlabel(xLabel)
    axis.set_ylabel("log(D) [mm]")
    axis.tick_params(labelleft = True) # show the y axis on both panels
    annotateFit(axis, fit)

figure.subplots_adjust(wspace = 0.28)
energyAxis.legend(loc = "upper left")
figure.suptitle("Crater Diameter vs. Kinetic Energy and Momentum")

figure.savefig("energy_momentum.png", dpi = 300, bbox_inches = "tight")
plt.close(figure)
