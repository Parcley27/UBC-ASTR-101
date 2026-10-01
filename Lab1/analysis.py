import math

import csv
import matplotlib.pyplot as plot

q = 43.3 # usable diagonal, mm

#l = 6.0 # physical focal length, mm, iPhone
#l35 = 52.0 # 35mm eq focal length (iPhone reported), mm

l = 75.0 # physical focal length, mm, canon camera
l35 = 75.0 # 35mm eq focal length, canon camera, mm

D = (q * l) / l35  # sensor usable diagonal, mm

#Nx = 3024.0 # image x pixel count, px, iPhone
#Ny = 4032.0 # image y pixel count, px, iPhone

Nx = 5184.0 # image x pixel count, px, canon camera
Ny = 3456.0 # image y pixel count, px, canon camera

w = D / (math.sqrt(Nx ** 2 + Ny ** 2)) # pixel width, mm/px

wOverride = 4.3e-3 # pixel width override for canon camera, mm/px
if wOverride is not None:
    w = wOverride

a = w / l # angular size/px, rad/px

object = "buoy"

with open(object + ".csv") as data:
    rows = list(csv.DictReader(data))

x1 = [float(row["baseline"]) for row in rows]
y1 = [a * float(row["px"]) for row in rows] # angular size, rad

# least-squares fit through the origin: y1 ~ m * x1
slope = sum(x * y for x, y in zip(x1, y1)) / sum(x ** 2 for x in x1)

distance = 1.0 / slope

print(f"Distance to {object}: {round(distance, 2)}m")
print(f"{distance}m")

# center data around the middle of the graph
def centeredUpper(values):
    return min(values) + max(values)

xMax = centeredUpper(x1)
yMax = centeredUpper(y1)

xFit = [0.0, xMax]
plot.plot(xFit, [slope * x for x in xFit], color = "#0b3d91", label = "Line of Best Fit", zorder = 1)

plot.scatter(x1, y1, color = "#FC3D21", label = "Observation Data", zorder = 2)

plot.xlabel("Baseline (meters)")
plot.ylabel("Angular Separation (radians)")
plot.title(f"Angular Separation of {object[0].upper() + object[1:]} vs. Baseline")

plot.axis((0.0, xMax, 0.0, yMax))

plot.annotate(f"Slope = {slope:.5g} rad/m", xy = (0.97, 0.03), xycoords = "axes fraction", ha = "right", va = "bottom")
plot.legend()

plot.savefig(f"{object}_graph.png", dpi = 300)
