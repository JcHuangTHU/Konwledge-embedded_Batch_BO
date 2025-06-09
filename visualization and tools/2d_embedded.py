"""
  Using UMAP to embed four-dimensional laser condition space into a two-dimensional plane for visualization.

  Output:
  ------
    file: umap_model.pkl, 2D-embedded model.
  ------
"""
import umap
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
import joblib

train_label = 0

x1 = np.linspace(0, 1, 55)
x2 = np.linspace(0, 1, 51)
x3 = np.linspace(0, 1, 8)
x4 = np.linspace(0, 1, 17)
hypercube = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)

if train_label:
    t0 = time.time()
    reducer = umap.UMAP(random_state=0)
    hypercube_2d = reducer.fit_transform(hypercube)

    joblib.dump(reducer, "umap_model.pkl")
    print("UMAP model has been saved")
    t1 = time.time()
    print("UMAP training consumes {} s".format(t1-t0))

else:
    reducer_loaded = joblib.load("umap_model.pkl")
    hypercube_2d = reducer_loaded.transform(hypercube)

hull = ConvexHull(hypercube_2d)
plt.figure(figsize=(8, 6))
for simplex in hull.simplices:
    plt.plot(hypercube_2d[simplex, 0], hypercube_2d[simplex, 1], 'r--', lw=1.5)

plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.title("Laser Conditions Projection based on UMAP")
plt.legend(["Laser Conditions 2d"], loc="best")
plt.show()



