import numpy as np
import matplotlib.pyplot as plt
from analyseOD import AnalysisData





# Paramètres de la gaussienne elliptique
x0, y0 = 50, 50  # Centre de la gaussienne
sigma_x, sigma_y = 20, 10  # Écart-types dans les directions x et y

# Création d'une grille d'indices
x = np.arange(0, 100)
y = np.arange(0, 100)
X, Y = np.meshgrid(x, y)

# Calcul de la gaussienne elliptique
Z = np.exp(-((X - x0) ** 2 / (2 * sigma_x ** 2) + (Y - y0) ** 2 / (2 * sigma_y ** 2)))

analysis = AnalysisData()

# Affichage de l'image
plt.imshow(Z, cmap='viridis')
plt.colorbar()
plt.title('Image à analyser : Gaussienne elliptique')
plt.show()

# Appel de la fonction d'analyse
handles = analyze(handles)
