import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import curve_fit

def gaussian_2d(xy, A0, x0, y0, sigma_x, sigma_y, C):
    x, y = xy
    return A0 * np.exp(-((x - x0)**2 / (2 * sigma_x**2)) - ((y - y0)**2 / (2 * sigma_y**2))) + C

def calculate_optical_density(image_transmitted, image_incident):
    # Convert images to numpy arrays
    I = np.array(image_transmitted, dtype=float)
    I0 = np.array(image_incident, dtype=float)
    
    # Avoid division by zero by replacing zero values with small values
    I0[I0 == 0] = 1e-6
    # Calculate optical density (OD)
    OD = np.zeros_like(I)  # Create an empty array with the same shape as I

    x_start, x_end = 500, 620
    y_start, y_end = 80, 425

    # Extract ROI from the optical density image
    roi = I[y_start:y_end, x_start:x_end]

    # Calculate mean value of the ROI
    mean_value = np.mean(roi)
    
    I = I/mean_value
    I0 = I0/mean_value

    # Calculate optical density (OD) while avoiding division by zero
    mask = (I != 0) & (I0 != 0)  # Mask to avoid division by zero
    OD[mask] = np.log10(I0[mask]/I[mask])  # Calculate OD with the mask
    
    # Replace NaNs with the maximum value of OD
    max_OD = np.nanmax(OD)
    OD[np.isnan(OD)] = max_OD
    
    return OD

# Load your own images instead of the dummy ones
image_transmitted = np.array(pd.read_csv('DATA/image.csv', delimiter=';'))
image_incident = np.array(pd.read_csv('DATA/bright.csv', delimiter=';'))

# Calculate optical density
optical_density = calculate_optical_density(image_transmitted, image_incident)



#optical_density = optical_density/mean_value

np.save("opticalDensity.npy", optical_density)
# Plot images and optical density with ROI
plt.figure(1)

plt.subplot(1, 2, 1)
plt.imshow(image_transmitted, cmap='gray')
plt.title('Transmitted Image')

plt.subplot(1, 2, 2)
plt.imshow(image_incident, cmap='gray')
plt.title('Incident Image')

# Plot ROI on optical density separately
plt.figure(2)
plt.imshow(optical_density, cmap='hot')
plt.colorbar(label='Optical Density (OD)')
plt.title('Optical Density with ROI')
plt.tight_layout()

plt.show()


