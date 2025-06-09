"""
  Visualization of the r-th iteration of R or GF optimization based on UMAP

  Parameters:
  ------
  R_Iter:
    Denotes the R_Iter iteration in the resistance optimization,
    where each iteration involves recommending a batch of experimental candidates,
    conducting physical measurements, and updating the observation dataset accordingly.

    * It can be changed to GF_Iter for constrained GF optimization.
  ------
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import joblib
from botorch.utils.transforms import normalize
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_data, x_norm_bd

R_Iter = 8

x1 = np.linspace(0, 1, 55)
x2 = np.linspace(0, 1, 51)
x3 = np.linspace(0, 1, 8)
x4 = np.linspace(0, 1, 17)
hypercube = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)
reducer_loaded = joblib.load("umap_model.pkl")
hypercube_2d = reducer_loaded.transform(hypercube)

hull = ConvexHull(hypercube_2d)

fig, ax = plt.subplots(figsize=(8, 6))
for simplex in hull.simplices:
    ax.plot(hypercube_2d[simplex, 0], hypercube_2d[simplex, 1], 'r--', lw=1.5, label="laser conditions 2d")

X, R, label, pre_R = get_data('../data/step2_Ropt/Ropt_iter{}.csv'.format(R_Iter - 1), name='Ropt')
x_norm = normalize(X, x_norm_bd)
x_norm_2d = reducer_loaded.transform(x_norm)

ax.scatter(x_norm_2d[:(R_Iter - 1) * 10, 0], x_norm_2d[:(R_Iter - 1) * 10, 1], color='blue', label=f"Observations",
           s=20)
ax.scatter(x_norm_2d[(R_Iter - 1) * 10:, 0], x_norm_2d[(R_Iter - 1) * 10:, 1], color='red', label=f"New suggestions",
           s=20,
           marker='*')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="best")
ax.set_xlabel("UMAP Dim 1")
ax.set_ylabel("UMAP Dim 2")
ax.set_title("Laser Conditions Projection based on UMAP")
plt.show()
