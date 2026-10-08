import numpy as np
import matplotlib.pyplot as plt

# Power law: P = C * a^x
# Linearize: log(P) = log(C) + x * log(a)  ->  y = intercept + slope * (log a), slope = x, intercept = log(C)

data = np.genfromtxt("observations.csv", delimiter = ",", skip_header = 1)
period = data[:, 1]  # orbital period (days)
semiMajorAxis = data[:, 2]  # semi-major axis (Jupiter radii)

logAxis = np.log10(semiMajorAxis)
logPeriod = np.log10(period)

slope, intercept = np.polyfit(logAxis, logPeriod, 1)
exponent = slope
coefficient = 10 ** intercept

print(f"slope (exponent x) = {exponent:.3f}")
print(f"intercept log(C)   = {intercept:.3f}")
print(f"C                  = {coefficient:.4f}")
print(f"P = {coefficient:.4f} * a^{exponent:.3f}")

# Plot the linearized data and fit
# extend the fit past the data on both sides, and far enough left to reach the y axis (log a = 0)
span = logAxis.max() - logAxis.min()
xLeft = min(0.0, logAxis.min()) - 0.15 * span
xRight = logAxis.max() + 0.15 * span
fitAxis = np.array([xLeft, xRight])

plt.plot(fitAxis, intercept + slope * fitAxis, color = "#0b3d91", label = "Line of Best Fit", zorder = 1)

plt.scatter(logAxis, logPeriod, color = "#FC3D21", label = "Observation Data", zorder = 2)

plt.scatter([0.0], [intercept], color = "#0b3d91", marker = "D", zorder = 3)
plt.annotate(f"y-intercept = {intercept:.3f}", xy = (0.0, intercept), xytext = (10, -4), textcoords = "offset points", ha = "left", va = "top")
plt.axvline(0.0, color = "gray", linewidth = 0.8, zorder = 0)
plt.axhline(0.0, color = "gray", linewidth = 0.8, zorder = 0)

plt.xlabel("log(a) [Jupiter radii]")
plt.ylabel("log(P) [days]")
plt.title("Orbital Period vs. Semi-Major Axis (log-log)")

plt.xlim(xLeft, xRight)
plt.annotate(f"Slope = {exponent:.3f}", xy = (0.97, 0.03), xycoords = "axes fraction", ha = "right", va = "bottom")
plt.legend(loc = "upper left")

plt.savefig("power_law_fit.png", dpi = 300)
