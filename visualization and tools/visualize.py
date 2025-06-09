"""
  This program is used to draw some key images, such as the visualization of cost model training, etc.
"""
from utils_visualize import *

plt_train_R_min('../data/step2_Ropt/Ropt_iter19.csv')
cost_model_visualize('../data/step3_GFopt/R_dataset.csv')
cost_model_visualize_heatmap('../data/step3_GFopt/R_dataset.csv')