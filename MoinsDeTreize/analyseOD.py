import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def density(OD,):
    roi_x_start, roi_x_end = 125, 350
    roi_y_start, roi_y_end = 375, 150

    # Extract ROI from the optical density image
    roi_OD = OD[roi_y_end:roi_y_start, roi_x_start:roi_x_end]
    cross_section = 2.907e-13   # [m^2]
    density = roi_OD/cross_section
    return density
    
    

# Load your image data

# Load your image data
# Replace 'opticalDensity' with your actual image array
image_array = np.load('opticalDensity.npy')  # Load your image array from file
roi_x_start, roi_x_end = 125, 350
roi_y_start, roi_y_end = 375, 150

# Extract ROI from the optical density image
roi_OD = image_array[roi_y_end:roi_y_start, roi_x_start:roi_x_end]
# Generate grid coordinates for the image
x = np.arange(image_array.shape[1])
y = np.arange(image_array.shape[0])
X, Y = np.meshgrid(x, y)

# Initial guess for the parameters (amplitude, xo, yo, sigma_x, sigma_y, theta)
initial_guess = (np.max(image_array), image_array.shape[1] / 2, image_array.shape[0] / 2, 400, 400, 0)

# Fit the Gaussian function to the image data
popt, pcov = curve_fit(gaussian_2d, (X, Y), image_array.ravel(), p0=initial_guess)

# Calculate standard deviation (uncertainty) for each fit parameter
residuals = image_array.ravel() - gaussian_2d((X, Y), *popt)
chi_squared = np.sum((residuals / image_array.ravel())**2)
N = len(image_array.ravel())
num_params = len(popt)
dof = N - num_params
reduced_chi_squared = chi_squared / dof
parameter_variances = np.diag(pcov)
parameter_uncertainties = np.sqrt(parameter_variances)

pixel_size_um = 8.46    # [um]
popt_um = np.array(popt) * pixel_size_um
parameter_uncertainties_um = parameter_uncertainties * pixel_size_um

rho_at = density(OD=image_array)
N_at = np.sum(rho_at*(8.46e-12))


# Plot original image
plt.figure(1)
plt.imshow(image_array, origin='lower', extent=(0, image_array.shape[1], 0, image_array.shape[0]), cmap='hot')
plt.title('Original Image')
plt.colorbar()
# Add crosshair at the center
plt.axhline(y=popt[2], color='r', linestyle='--', linewidth=1)
plt.axvline(x=popt[1], color='r', linestyle='--', linewidth=1)
plt.tight_layout()


# Plot fitted Gaussian profile
plt.figure(2)
fitted_gaussian = gaussian_2d((X, Y), *popt).reshape(image_array.shape)
plt.imshow(fitted_gaussian, origin='lower', extent=(0, image_array.shape[1], 0, image_array.shape[0]), cmap='hot')
plt.title(r'Fitted Gaussian Profile, $N_{at}$' f'$ = {np.round(N_at*1e-6, 2)} M$')
plt.colorbar()
roi_x_start, roi_x_end = 125, 350
roi_y_start, roi_y_end = 375, 150
# Add crosshair at the center
plt.axhline(y=popt[2], color='r', linestyle='--', linewidth=1)
plt.axvline(x=popt[1], color='r', linestyle='--', linewidth=1)
plt.axhline(y=roi_y_start, color='w', linestyle='--', linewidth=1)
plt.axhline(y=roi_y_end, color='w', linestyle='--', linewidth=1)
plt.axvline(x=roi_x_start, color='w', linestyle='--', linewidth=1)
plt.axvline(x=roi_x_end, color='w', linestyle='--', linewidth=1)
# Add legend for parameter uncertainties
plt.text(10, 10, f"Amplitude: {popt[0]:.2f} ± {parameter_uncertainties[0]:.2f}", color='white', fontsize=12)
plt.text(10, 30, f"x0: {popt_um[1]:.2f} ± {parameter_uncertainties_um[1]:.2f} μm", color='white', fontsize=12)
plt.text(10, 50, f"y0: {popt_um[2]:.2f} ± {parameter_uncertainties_um[2]:.2f} μm", color='white', fontsize=12)
plt.text(10, 70, f"Sigma_x: {popt_um[3]:.2f} ± {parameter_uncertainties_um[3]:.2f} μm", color='white', fontsize=12)
plt.text(10, 90, f"Sigma_y: {popt_um[4]:.2f} ± {parameter_uncertainties_um[4]:.2f} μm", color='white', fontsize=12)
plt.text(10, 110, f"Theta: {popt_um[5]:.2f} ± {parameter_uncertainties[5]:.2f} °", color='white', fontsize=12)
plt.tight_layout()

plt.figure(3)
plt.plot(image_array[:, int(popt[1])], label='Raw Data (X direction)')
plt.plot(fitted_gaussian[:, int(popt[1])], label='Fitted Data (X direction)')
plt.title('Raw and Fitted Data (X direction)')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()

plt.figure(4)
plt.plot(image_array[int(popt[2]), :], label='Raw Data (Y direction)')
plt.plot(fitted_gaussian[int(popt[2]), :], label='Fitted Data (Y direction)')
plt.title('Raw and Fitted Data (Y direction)')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()