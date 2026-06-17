import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from skimage.feature import graycomatrix, graycoprops

Tk().withdraw()  

image_path = askopenfilename(
    title="Select an Image for Lab 7 Texture Analysis",
    filetypes=[
        ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
    ]
)

if image_path == "":
    print("No image selected! Exiting program...")
    exit()

print("Selected image:", image_path)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not decode or read the file at layout path: {image_path}")
    exit()

glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=False, normed=True)

glcm_2d = glcm[:, :, 0, 0]

max_prob = np.max(glcm_2d)
contrast = graycoprops(glcm, 'contrast')[0, 0]
uniformity = graycoprops(glcm, 'ASM')[0, 0]  
correlation = graycoprops(glcm, 'correlation')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

nonzero_elements = glcm_2d[glcm_2d > 0]
entropy = -np.sum(nonzero_elements * np.log2(nonzero_elements))

filename = os.path.basename(image_path)
results_text = (
    f"--- Results for: {filename} ---\n"
    f"Maximum Probability: {max_prob:.5f}\n"
    f"Contrast: {contrast:.2f}\n"
    f"Uniformity (ASM): {uniformity:.5f}\n"
    f"Correlation: {correlation:.4f}\n"
    f"Homogeneity: {homogeneity:.4f}\n"
    f"Entropy: {entropy:.2f}"
)
print("\n" + results_text)

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f"Target Surface: {filename}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.log1p(glcm_2d), cmap='magma', extent=[0, 255, 255, 0])
plt.title("GLCM Structural Matrix (Log Scaled)")
plt.colorbar(label='Log-scaled Probability intensity')
plt.xlabel("Neighbor Pixel Intensity (Grey Level)")
plt.ylabel("Reference Pixel Intensity (Grey Level)")

plt.gcf().text(0.15, 0.02, results_text, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.6", facecolor="azure", edgecolor="cadetblue", alpha=0.9))

plt.tight_layout()
plt.show()