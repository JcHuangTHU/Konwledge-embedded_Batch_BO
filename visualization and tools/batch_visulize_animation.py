"""
  UMAP-based visualization of R or constrained GF optimization from the first r rounds.

  Parameters:
  ------
  GF_Iter:
    Denotes the GF_Iter iteration in the constrained GF optimization,
    where each iteration involves recommending a batch of experimental candidates,
    conducting physical measurements, and updating the observation dataset accordingly.

    * It can be changed to R_Iter for R optimization.
  ------

  Output:
  ------
    A gif format animation.
  ------
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import joblib
from matplotlib.animation import FuncAnimation
from botorch.utils.transforms import normalize
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_data, x_norm_bd

GF_Iter = 7

x1 = np.linspace(0, 1, 55)
x2 = np.linspace(0, 1, 51)
x3 = np.linspace(0, 1, 8)
x4 = np.linspace(0, 1, 17)
hypercube = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)
reducer_loaded = joblib.load("umap_model.pkl")
hypercube_2d = reducer_loaded.transform(hypercube)

hull = ConvexHull(hypercube_2d)

X, R, label, pre_R = get_data('../data/step3_GFopt/GF_{}.csv'.format(GF_Iter - 1), name='GFopt')
x_norm = normalize(X, x_norm_bd)
x_norm_2d = reducer_loaded.transform(x_norm)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

fig, ax = plt.subplots(figsize=(8, 6))
for simplex in hull.simplices:
    ax.plot(hypercube_2d[simplex, 0], hypercube_2d[simplex, 1], 'r--', lw=1.5, label="laser conditions boundary")

scatter_prev = ax.scatter([], [], c='blue', alpha=0.7, s=30, label="Observations")
scatter_new = ax.scatter([], [], c='red', marker='*', s=80, label="New suggestions")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)


def update(frame):
    obs = x_norm_2d[:frame * 8]  # 之前的数据
    suggestions = x_norm_2d[frame * 8: (frame + 1) * 8]
    scatter_prev.set_offsets(obs)
    scatter_new.set_offsets(suggestions)
    ax.set_title(f"Iteration {frame}")
    return scatter_prev, scatter_new


ani = FuncAnimation(fig, update, frames=GF_Iter, interval=1000, repeat=True)

ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
# ax.set_title("Gauge Factor Optimization Process")

ani.save(f"GF_batch_suggestions_Iter{GF_Iter}.gif", writer="imagemagick")

plt.show()
