import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# Schematic cross-section of the drop setup (arbitrary units, not to scale)

fitColor = "#0b3d91"
dataColor = "#FC3D21"
densityColor = "#1B998B"
velocityColor = "#F2A900"
wallColor = "#333333"
ballColor = "#9AA0A6" # grey

# geometry
boxLeft = 0.0
boxRight = 8.8 # wide enough for the right-hand labels
floorY = 0.0
wallTop = 3.0
sandSurface = 1.6 # d

ballX = 3.6
hX = 0.8 # x of the h marker
stackX = 5.8 # x of the d and h - d markers
ballRadius = 0.45
ballBottom = 6.0

craterCenter = ballX # crater directly under the ball
craterRadius = 1.2 # half the rim-to-rim distance
bowlDepth = 0.75
rimHeight = 0.25
rimWidth = 0.35

# Drop height is taken from the box floor to the bottom of the ball (unconfirmed with the group).
# To change the convention, change dropReferenceY / dropTopY and the labels below.
dropReferenceY = floorY
dropTopY = ballBottom

dropLabel = "h (Drop Height)"
depthLabel = "d (Sand Depth)"
fallLabel = "h − d (Fall Distance)"
craterLabel = "D"


def sandProfile(x):
    offset = np.abs(x - craterCenter)
    bowl = bowlDepth * np.clip(1 - (offset / craterRadius) ** 2, 0, None)
    rim = rimHeight * np.exp(-((offset - craterRadius) / rimWidth) ** 2)
    return sandSurface - bowl + rim


arrowWidth = 2.5 # line width, points
arrowGap = 5 # space between arrow heads and the lines they point at, points
dotted = (0, (1, 2)) # one dotted style for every guide line
guideWidth = 1.5


def doubleArrow(axis, start, end, color):
    axis.annotate("", xy = end, xytext = start,
                  arrowprops = dict(arrowstyle = "|-|,widthA = 0.6,widthB = 0.6", color = color, linewidth = arrowWidth,
                                    mutation_scale = 18, shrinkA = arrowGap, shrinkB = arrowGap))


figure, axis = plt.subplots(figsize = (8, 6))

# sand, with the crater cut into its surface
xSurface = np.linspace(boxLeft, boxRight, 400)
ySurface = sandProfile(xSurface)
sandOutline = np.column_stack([np.concatenate([xSurface, [boxRight, boxLeft]]), np.concatenate([ySurface, [floorY, floorY]])])
axis.add_patch(Polygon(sandOutline, closed = True, facecolor = velocityColor, alpha = 0.35, edgecolor = "none"))
axis.plot(xSurface, ySurface, color = velocityColor, linewidth = 1.5)

# box floor and walls
axis.plot([boxLeft, boxLeft, boxRight, boxRight], [wallTop, floorY, floorY, wallTop], color = wallColor, linewidth = 3, solid_capstyle = "butt")

# ball
axis.add_patch(Circle((ballX, ballBottom + ballRadius), ballRadius, facecolor = ballColor, edgecolor = wallColor, zorder = 3))
axis.annotate("", xy = (ballX, ballBottom - 1.0), xytext = (ballX, ballBottom),
              arrowprops = dict(arrowstyle = "->,head_length = 0.5,head_width = 0.25", color = ballColor, linewidth = arrowWidth,
                                mutation_scale = 18, shrinkA = arrowGap, shrinkB = 0))

# guide lines at the ball bottom and the sand surface
axis.plot([hX, stackX], [ballBottom, ballBottom], color = "gray", linestyle = dotted, linewidth = guideWidth)

# h: floor to ball bottom
doubleArrow(axis, (hX, dropReferenceY), (hX, dropTopY), fitColor)
axis.text(hX - 0.15, (dropReferenceY + dropTopY) / 2, dropLabel, rotation = 90, ha = "right", va = "center", color = fitColor)

# d and h - d, stacked at the same x
doubleArrow(axis, (stackX, floorY), (stackX, sandSurface), densityColor)
axis.text(stackX + 0.15, (floorY + sandSurface) / 2, depthLabel, ha = "left", va = "center", color = densityColor)

doubleArrow(axis, (stackX, sandSurface), (stackX, ballBottom), fitColor)
axis.text(stackX + 0.15, (sandSurface + ballBottom) / 2, fallLabel, ha = "left", va = "center", color = fitColor)

# D: between the rim crests
leftSide = xSurface < craterCenter
rightSide = xSurface > craterCenter
leftCrest = xSurface[leftSide][np.argmax(ySurface[leftSide])]
rightCrest = xSurface[rightSide][np.argmax(ySurface[rightSide])]
crestY = ySurface.max()
arrowY = crestY + 0.45

for crestX in (leftCrest, rightCrest):
    axis.plot([crestX, crestX], [crestY, arrowY + 0.1], color = "gray", linestyle = dotted, linewidth = guideWidth)
doubleArrow(axis, (leftCrest, arrowY), (rightCrest, arrowY), dataColor)
axis.text(craterCenter, arrowY + 0.1, craterLabel, ha = "center", va = "bottom", color = dataColor, fontsize = 12)

axis.set_xlim(-0.6, boxRight + 0.4)
axis.set_ylim(-0.4, 7.2)
axis.set_aspect("equal")
axis.axis("off")

figure.savefig("setup_diagram.png", dpi = 300, bbox_inches = "tight")
plt.close(figure)
