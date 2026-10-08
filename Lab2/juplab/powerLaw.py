import numpy as np
import matplotlib.pyplot as plt

# Power law: P = C * a^x
# Linearize: log(P) = log(C) + x * log(a)  ->  y = intercept + slope * (log a), slope = x, intercept = log(C)

data = np.genfromtxt("observations.csv", delimiter=",", skip_header=1)
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
fitAxis = np.linspace(logAxis.min(), logAxis.max(), 100)

plt.scatter(logAxis, logPeriod, label="observations")
plt.plot(fitAxis, intercept + slope * fitAxis, label=f"fit: slope = {exponent:.2f}")
plt.xlabel("log(a) [Jupiter radii]")
plt.ylabel("log(P) [days]")
plt.legend()
plt.savefig("power_law_fit.png", dpi=150)
plt.show()
