import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


def gaussian_lowpass(img, sigma=1.0):
    return gaussian_filter(img.astype(np.float64), sigma=sigma)


def downsample(img):
    return img[::2, ::2]


def upsample_bilinear(img, target_shape):
    zoom_r = target_shape[0] / img.shape[0]
    zoom_c = target_shape[1] / img.shape[1]
    return zoom(img, (zoom_r, zoom_c), order=1)  


def build_pyramids(image, p=3, sigma=1.0):
    approx_pyr   = [image.copy()]   
    residual_pyr = []

    current = image.copy()

    for _ in range(p):
        filtered    = gaussian_lowpass(current, sigma=sigma)
        downsampled = downsample(filtered)

        predicted = upsample_bilinear(downsampled, current.shape)

        residual = current.astype(np.float64) - predicted

        approx_pyr.append(downsampled)
        residual_pyr.append(residual)

        current = downsampled   

    residual_pyr.append(approx_pyr[-1])

    return approx_pyr, residual_pyr


def visualise(approx_pyr, residual_pyr, p):
    n_approx = len(approx_pyr)   

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Image Pyramid  (p = {})".format(p), fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(2, n_approx, figure=fig, hspace=0.45, wspace=0.3)

    for i, img in enumerate(approx_pyr):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        level_label = "Level J" if i == 0 else "Level J-{}".format(i)
        ax.set_title("Approx\n{}\n{}x{}".format(level_label, img.shape[0], img.shape[1]),
                     fontsize=8)
        ax.axis('off')

    for i, img in enumerate(residual_pyr):
        ax = fig.add_subplot(gs[1, i])
        if i < p: 
            display = np.clip(img + 128, 0, 255)
            ax.imshow(display, cmap='gray', vmin=0, vmax=255)
            level_label = "Level J" if i == 0 else "Level J-{}".format(i)
            title = "Residual\n{}\n{}x{}".format(level_label, img.shape[0], img.shape[1])
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            title = "Apex Approx\n(Residual apex)\n{}x{}".format(img.shape[0], img.shape[1])
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    out_path = "pyramid_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Saved visualisation ->", out_path)
    plt.close()
    return out_path


def main():
    image = np.array(Image.open("messi.jpeg").convert('L'), dtype=np.float64)

    p     = 3
    sigma = 1.0

    print(f"Building pyramid with p={p}, Gaussian sigma={sigma}, bilinear upsampling ...")
    approx_pyr, residual_pyr = build_pyramids(image, p=p, sigma=sigma)

    print("\nApproximation Pyramid:")
    for i, a in enumerate(approx_pyr):
        tag = "original" if i == 0 else f"downsampled x{2**i}"
        print(f"  Level J-{i:>1}  {a.shape[0]:>4}x{a.shape[1]:<4}  ({tag})")

    print("\nPrediction Residual Pyramid:")
    for i, r in enumerate(residual_pyr):
        tag = "apex approx" if i == p else "residual"
        print(f"  Level J-{i:>1}  {r.shape[0]:>4}x{r.shape[1]:<4}  ({tag})")

    total = sum(a.size for a in approx_pyr[:-1]) + approx_pyr[-1].size
    N2    = image.size
    print(f"\nTotal pixels in pyramid : {total}")
    print(f"Original pixels (N^2)   : {N2}")
    print(f"Ratio                   : {total/N2:.4f}  (should be < 4/3 = 1.3333)")

    out = visualise(approx_pyr, residual_pyr, p)
    print("\nDone! Output saved to:", out)


if __name__ == "__main__":
    main()
