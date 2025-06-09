"""
  The first step is to use Latin hypercube sampling for
  initial sampling from four-dimensional laser conditions.
"""
import pandas as pd
import numpy as np
from scipy.stats import qmc


def map_bound(lb_, sp_):
    lb_map_ = np.empty((0, 2), float)
    for i_ in range(len(lb_)):
        lb_map_ = np.vstack((lb_map_, lb_[i_] / sp_[i_]))
    sp_map = np.array([1, 1, 1, 1])
    return lb_map_, sp_map


def init_lhs(dim_, num_, lb, sp):
    sampler = qmc.LatinHypercube(d=dim_, seed=0)
    sample = sampler.random(n=num_)
    sample_scaled = qmc.scale(sample, lb[:, 0], lb[:, 1])
    sample_scaled = np.round(sample_scaled, decimals=0)
    sample_remap = sample_scaled * sp
    sam_pd = pd.DataFrame(sample_remap, index=np.arange(1, num_ + 1, 1),
                          columns=['power (%)', 'df (mm)', 'frequency (kHz)', 'speed (mm/s)'])
    sam_pd.index.name = 'Number'
    sam_pd.to_csv('./data/step1_init/lhs_init.csv')
    print(sam_pd)


# laser space: power step=0.5, defocus step=0.1, frequency step=50, speed step=5
laser_bound = np.array([[3, 30], [0, 5], [150, 500], [20, 100]])
spacing = np.array([0.5, 0.1, 50, 5])
laser_bound_map, spacing_map = map_bound(laser_bound, spacing)

init_lhs(dim_=4, num_=10, lb=laser_bound_map, sp=spacing)
