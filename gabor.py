import numpy as np
import matplotlib.pyplot as plt

def gabor_wavelet(size, theta, wavelength, sigma, aspect_ratio):
    """
    Generate a Gabor wavelet filter.

    Parameters:
        size (int): Size of the filter (size x size).
        theta (float): Orientation in radians.
        wavelength (float): Wavelength of the sinusoidal component.
        sigma (float): Standard deviation of the Gaussian envelope.
        aspect_ratio (float): Ratio of the minor axis to the major axis of the Gaussian envelope.

    Returns:
        np.ndarray: The generated Gabor wavelet filter.
    """
    # Create a grid of (x, y) coordinates
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)

    # Rotate coordinates by theta
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor wavelet formula
    gaussian_envelope = np.exp(-0.5 * (x_theta**2 + (aspect_ratio * y_theta)**2) / sigma**2)
    sinusoidal_wave = np.cos(2 * np.pi * x_theta / wavelength)
    gabor = gaussian_envelope * sinusoidal_wave

    return gabor

# Parameters for the Gabor wavelet filters
size = 51
orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0, 45, 90, 135 degrees
wavelength = 10
sigma = 8
aspect_ratio = 0.5

# Generate and plot the Gabor wavelets
fig, axes = plt.subplots(1, len(orientations), figsize=(15, 4))
for i, theta in enumerate(orientations):
    gabor = gabor_wavelet(size, theta, wavelength, sigma, aspect_ratio)
    axes[i].imshow(gabor, cmap='gray', extent=(-size//2, size//2, -size//2, size//2))
    axes[i].set_title(f"Theta = {theta * 180 / np.pi:.1f}°")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
