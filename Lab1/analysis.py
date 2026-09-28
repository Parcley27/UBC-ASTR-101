import math

import csv
import matplotlib.pyplot as plot

q = 43.3 # usable diagonal, mm
l = 6.0 # physical focal length, mm
l35 = 52.0 # 35mm eq focal length (iphone reported), mm

D = (q * l) / l35  # sensor usable diagonal, mm

Nx = 3024.0 # image x pixel count, px
Ny = 4032.0 # image y pixel count, px

w = D / (math.sqrt(Nx ** 2 + Ny ** 2)) # pixel width, mm/px

a = w / l # angular size/px, rad/px

with open("images.csv") as data:
    rows = list(csv.DictReader(data))

x1 = [float(row["baseline"]) for row in rows]
y1 = [a * float(row["px"]) for row in rows] # angular size, rad

# least-squares fit through the origin: y1 ~ m * x1
slope = sum(x * y for x, y in zip(x1, y1)) / sum(x ** 2 for x in x1)

print(slope)

xFit = [0.0, 7.0] # from the origin to the right edge of the plot
plot.plot(xFit, [slope * x for x in xFit], color = "#0b3d91", label = "Line of Best Fit", zorder = 1)

plot.scatter(x1, y1, color = "#FC3D21", label = "Observation Data", zorder = 2)

plot.xlabel("Baseline (meters)")
plot.ylabel("Angular Size (radians)")

plot.axis((0.0, 7.0, 0.0, 0.025))

plot.legend()

plot.savefig("graph.png", dpi = 300)
