import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

class AnalyseOD:
    def __init__(self, images, analys_ROI=None, normalization_ROI=None) -> None:
        self.dark_image = np.array(images[0], dtype=float)
        self.bright_image = np.array(images[1], dtype=float)
        self.analys_ROI = analys_ROI
        self.normalization_ROI = normalization_ROI

    def calculate_OD(self):
        x_start, x_end = self.normalization_ROI[0], self.normalization_ROI[1]
        y_start, y_end = self.normalization_ROI[2], self.normalization_ROI[3]
    
        roi = self.dark_image[y_start:y_end, x_start: x_end]
        mean_value = np.mean(roi)
        normalized_dark_image = self.dark_image/mean_value
        normalized_bright_image = self.bright_image/mean_value
        self.dark_image[self.dark_image == 0] = 1e-6
        mask = (normalized_dark_image != 0) & (normalized_bright_image != 0)

        optical_density = np.zeros_like(self.dark_image)
        optical_density[mask] = np.log10(normalized_bright_image[mask]/normalized_dark_image[mask])

        max_OD = np.amax(optical_density) 
        optical_density[optical_density == 0] = max_OD

        return optical_density
    
    def fit_cloud_profile_gaussian1D(self, OD):

        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)
        
        roi_x_start, roi_x_end = self.analys_ROI[0], self.analys_ROI[1]
        roi_y_start, roi_y_end = self.analys_ROI[2], self.analys_ROI[3]
        roi_OD = OD[roi_y_end:roi_y_start, roi_x_start:roi_x_end]
        
        # grid for the image 
        x = np.arange(OD.shape[1])
        y = np.arange(OD.shape[0])
        print(roi_x_end-roi_x_start)
        OD_x = OD[int(roi_x_end-roi_x_start), :]
        OD_y = OD[:,int(roi_y_start-roi_y_end)]
        plt.figure(6)
        plt.plot(OD_x)
        plt.plot(OD_y)
        # fit the profiles of the cloud in the ROI. 
        initial_guess_x = (np.amax(OD_x), np.mean(OD_x), np.std(OD_x))
        initial_guess_y = (np.amax(OD_y), np.mean(OD_y), np.std(OD_y))
        poptx, pcovx = curve_fit(gaussian, x, OD_x, p0=initial_guess_x)
        popty, pcovy = curve_fit(gaussian, y, OD_y, p0=initial_guess_y)
        fitted_profile_x = gaussian(x, *poptx)   # to return 
        fitted_profile_y = gaussian(y, *popty)

        # Calculate standard deviation (uncertainty) for each fit parameter
        def determine_uncertainty(popt, pcov, x, OD_prof):
            residuals = OD_prof - gaussian(x, *popt)
            OD_prof[OD_prof == 0] = 1e-6
            chi_squared = np.sum((residuals /OD_prof)**2)
            N = len(OD_prof)
            num_params = len(popt)
            dof = N - num_params
            reduced_chi_squared = chi_squared / dof
            parameter_variances = np.diag(pcov)
            return np.sqrt(parameter_variances)  # to return [not good dimension yet]
        
        parameters_uncertainty_x = determine_uncertainty(poptx, pcovx, x, OD_prof=OD[int((roi_y_end-roi_y_start)/2), :] )
        parameters_uncertainty_y = determine_uncertainty(popty, pcovy, y, OD_prof=OD[:,int((roi_x_end-roi_x_start)/2)])
        return  fitted_profile_x, fitted_profile_y, parameters_uncertainty_x, parameters_uncertainty_y

    def fit_cloud_profile_gaussian2D(self, OD):

        def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
            x, y = xy
            xo = float(xo)
            yo = float(yo)
            a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
            b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
            c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
            g = amplitude * np.exp(- (a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
            return g.ravel()
        
        roi_x_start, roi_x_end = self.analys_ROI[0], self.analys_ROI[1]
        roi_y_start, roi_y_end = self.analys_ROI[2], self.analys_ROI[3]
        roi_OD = OD[roi_y_end:roi_y_start, roi_x_start:roi_x_end]

        # grid for the image 
        x = np.arange(OD.shape[1])
        y = np.arange(OD.shape[0])
        X, Y = np.meshgrid(x, y)


        # Calculate the number of atoms in the ROI of analysis 
        cross_section = 2.907e-13   # [m^2]
        density = roi_OD/cross_section
        number_atoms = np.sum(density*(8.46e-6)**2) # to return 

        # fit the profile of the cloud in the ROI. 
        initial_guess = (np.max(OD), OD.shape[1] / 2, OD.shape[0] / 2, 400, 400, 0)
        popt, pcov = curve_fit(gaussian_2d, (X, Y), OD.ravel(), p0=initial_guess)
        fitted_profile = gaussian_2d((X, Y), *popt).reshape(OD.shape)   # to return 

        # Calculate standard deviation (uncertainty) for each fit parameter
        residuals = OD.ravel() - gaussian_2d((X, Y), *popt)
        OD[OD == 0] = 1e-6
        chi_squared = np.sum((residuals / OD.ravel())**2)
        N = len(OD.ravel())
        num_params = len(popt)
        dof = N - num_params
        reduced_chi_squared = chi_squared / dof
        parameter_variances = np.diag(pcov)
        parameters_uncertainty = np.sqrt(parameter_variances)  # to return [not good dimension yet]
        
        return number_atoms, fitted_profile, parameters_uncertainty



image_transmitted = np.array(pd.read_csv('DATA/image.csv', delimiter=';'))
image_incident = np.array(pd.read_csv('DATA/bright.csv', delimiter=';'))
images = [image_transmitted, image_incident]
a_ROI = [100, 350, 375, 150]
n_ROI = [500, 620, 80, 425]
aod = AnalyseOD(images=images, analys_ROI=a_ROI, normalization_ROI=n_ROI)
OD = aod.calculate_OD()
number_atoms, fitted_profile, parameters_uncertainty = aod.fit_cloud_profile_gaussian2D(OD=OD)

H_cut, V_cut, hcut_param_uncertainty, vcut_param_uncertainty = aod.fit_cloud_profile_gaussian1D(OD=OD)

plt.figure(1)
plt.imshow(OD, cmap='hot')
plt.colorbar(label='Optical Density (OD)')
plt.title('Optical Density with ROI')
plt.tight_layout()

plt.figure(2)
plt.title(r'Gaussian 2D Profile, $N_{at}$' f'$ = {np.round(number_atoms*1e-6, 2)} M$')
plt.imshow(fitted_profile, cmap='hot')
plt.colorbar(label='Optical Density (OD)')
plt.tight_layout()

plt.figure(3)
plt.plot(V_cut)

plt.figure(4)
plt.plot(H_cut)
plt.show()
