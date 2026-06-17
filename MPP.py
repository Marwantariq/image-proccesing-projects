import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
image_path = askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif")
    ]
)

if image_path == "":
    print("No image selected!")
    exit()

print("Selected image:", image_path)

img = cv2.imread(image_path, 0)

if img is None:
    print("Error loading image!")
    exit()

blur = cv2.GaussianBlur(img, (5, 5), 0)

_, binary = cv2.threshold(
    blur,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = np.ones((5, 5), np.uint8)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)

boundary = max(contours, key=cv2.contourArea)

boundary_points = boundary[:, 0, :]

def orientation(a, b, c):

    return (
        a[0] * (b[1] - c[1])
        - a[1] * (b[0] - c[0])
        + (b[0] * c[1] - c[0] * b[1])
    )

def resample_boundary(points, cell_size):

    sampled = []

    for p in points:

        x = int(round(p[0] / cell_size) * cell_size)
        y = int(round(p[1] / cell_size) * cell_size)

        sampled.append((x, y))

    unique = []

    for p in sampled:
        if len(unique) == 0 or p != unique[-1]:
            unique.append(p)

    return np.array(unique)


def classify_vertices(points):

    white_vertices = []
    black_vertices = []

    n = len(points)

    for i in range(n):

        prev = points[(i - 1) % n]
        curr = points[i]
        nxt = points[(i + 1) % n]

        det = orientation(prev, curr, nxt)

        if det > 0:
            white_vertices.append(curr)

        elif det < 0:
            black_vertices.append(curr)

    return np.array(white_vertices), np.array(black_vertices)


def mirror_black_vertices(black_vertices, cell_size):

    mirrored = []

    for p in black_vertices:

        x, y = p

        mirrored.append((x + cell_size, y + cell_size))

    return np.array(mirrored)

def mpp_polygon(white_vertices, mirrored_black):

    all_points = []

    for p in white_vertices:
        all_points.append(tuple(p))

    for p in mirrored_black:
        all_points.append(tuple(p))

    all_points = np.array(all_points, dtype=np.float32)

    if len(all_points) < 3:
        return all_points

    epsilon = 0.02 * cv2.arcLength(all_points, True)

    polygon = cv2.approxPolyDP(
        all_points,
        epsilon,
        True
    )

    return polygon[:, 0, :]


cell_sizes = [2, 3, 4, 6, 8, 16, 32]


fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for idx, cell in enumerate(cell_sizes):

    sampled = resample_boundary(
        boundary_points,
        cell
    )

    white_v, black_v = classify_vertices(sampled)

    mirrored_black = mirror_black_vertices(
        black_v,
        cell
    )

    polygon = mpp_polygon(
        white_v,
        mirrored_black
    )

    ax = axes[idx]

    ax.imshow(binary, cmap='gray')

    ax.plot(
        sampled[:, 0],
        sampled[:, 1],
        'b.',
        markersize=1
    )

    if len(polygon) > 0:

        poly = np.vstack([polygon, polygon[0]])

        ax.plot(
            poly[:, 0],
            poly[:, 1],
            'r-',
            linewidth=2
        )

    ax.set_title(f"Cell Size = {cell}")

    ax.invert_yaxis()

    ax.axis('off')

fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()