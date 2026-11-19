import math
import sys

from craterData import gravity, loadImpacts

sys.stdout.reconfigure(encoding = "utf-8")

# How much does air drag lower the impact speed below sqrt(2 g h_f)?
# Quadratic drag from rest: v^2 = (g / k)(1 - exp(-2 k y)), with k = rho_air * C_d * A / (2 m)

airDensity = 1.2 # kg/m^3
dragCoefficient = 0.47 # sphere
gravitySi = gravity / 100.0 # m/s^2

# (label, impact number): the lightest balls and the extremes of size and fall height
cases = [
    ("Nylon 1.27 cm", 10),
    ("Teflon 1.27 cm", 9),
    ("Steel 0.476 cm", 1),
    ("Steel 1.27 cm", 16),
    ("Steel 2.54 cm", 6),
]

data = loadImpacts()


def impactSpeeds(mass, diameter, fallHeight):
    """Speeds in m/s with and without drag, for SI inputs (kg, m, m)."""
    area = math.pi * (diameter / 2) ** 2 # m^2
    k = airDensity * dragCoefficient * area / (2 * mass) # 1/m

    withDrag = math.sqrt(gravitySi / k * (1 - math.exp(-2 * k * fallHeight)))
    withoutDrag = math.sqrt(2 * gravitySi * fallHeight)
    return withDrag, withoutDrag


print("case             fall (cm)  v drag (m/s)  v no drag (m/s)  change")
for label, impactNumber in cases:
    index = list(data["impact"]).index(impactNumber)

    mass = data["mass"][index] / 1000.0 # kg
    diameter = data["ballDiameter"][index] / 100.0 # m
    fallHeight = data["fallHeight"][index] / 100.0 # m

    withDrag, withoutDrag = impactSpeeds(mass, diameter, fallHeight)
    change = (withDrag / withoutDrag - 1) * 100 # %

    print(f"{label:15s}  {fallHeight * 100:9.1f}  {withDrag:12.4f}  {withoutDrag:15.4f}  {change:+.2f}%")
