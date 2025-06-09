"""
  This program is based on the Knowledge-embedded batch BO to recommend next batch of experimental points,
  in order to find the laser conditions corresponding to high average GF under resistance constraints

  Parameters:
  ------
  GF_Iter:
    Denotes the GF_Iter iteration in the constrained GF optimization,
    where each iteration involves recommending a batch of experimental candidates,
    conducting physical measurements, and updating the observation dataset accordingly.

  next_num:
    The batch size of the recommended experimental points.

  scale_num:
    The number of length-scale values uniformly sampled in logarithmic space between 1e−3 and 1e3,
    used to construct a batch of Gaussian process (GP) models.
    Each sampled length-scale corresponds to an individual GP model,
    enabling multi-scale modeling of the objective function.
  ------
"""
from utils import *
from gpytorch.kernels import MaternKernel, RBFKernel
from sklearn_extra.cluster import KMedoids

GF_Iter = 1
next_num = 8  # Number of recommendations for each iter
scale_num = 20  # Scale of each multi-scale model
random_seed = 1
np.random.seed(random_seed)

# get R dataset
cost_X, R, label, _ = get_data('./data/step3_GFopt/R_dataset.csv', name='Ropt')
cost_x = normalize(cost_X, x_norm_bd)
cost_y = normalize(R, y_norm_bd_R)

cost_model, cost_likelihood = fit_gp_model(cost_x, cost_y, name='Iter {}'.format(GF_Iter), num_train_iters=500)
print_model_info(cost_model)

obj_X, avg_GF, pre_R, pre_GF = get_data('./data/step3_GFopt/GF_{}.csv'.format(GF_Iter - 1), name='GFopt')
obj_x = normalize(obj_X, x_norm_bd)
obj_y = normalize(avg_GF, y_norm_bd_GF)
best_y = obj_y.max()

decision_maker = Multiscale_BO_decision_maker(obj_x, obj_y, best_y)
decision_maker.get_search_batch()
decision_maker.get_infeasible_domain(label, X=cost_X)

cand_list = torch.empty(0, 4)
y_pre_list = torch.empty(0, 1)
kernels = [MaternKernel(nu=0.5), MaternKernel(nu=1.5), MaternKernel(nu=2.5), RBFKernel()]
length_scale_array = 10 ** (np.linspace(-3, 3, scale_num))
np.random.shuffle(length_scale_array)

for kernel in kernels:
    decision_maker.register_model(kernel)
    for i in length_scale_array:
        kernel_i = kernel
        kernel_i.lengthscale = i
        decision_maker.model_step(kernel_i)

        cand, y_pre = decision_maker.recommend(policy='cei', maximize=True, cost_model=cost_model)

        if cand is not None:
            cand_list = torch.cat([cand_list, cand[None, :]], dim=0)
            y_pre_list = torch.cat([y_pre_list, y_pre[None, :]], dim=0)

cand_list = np.array(cand_list.detach().numpy())
y_pre_list = np.array(y_pre_list.detach().numpy())
candidates, idx = np.unique(cand_list, axis=0, return_index=True)
y_pre_list = y_pre_list[idx]
norm_cand = normalize(torch.tensor(candidates), x_norm_bd)
norm_y_pre = normalize(torch.tensor(y_pre_list), y_norm_bd_GF)

kmedoids = KMedoids(n_clusters=next_num,
                    method='pam',
                    init='k-medoids++',
                    max_iter=500,
                    random_state=1).fit(norm_cand)

kmedoids_visual(norm_cand, norm_y_pre, kmedoids, name='GF optimize: Iter {}'.format(GF_Iter))
kmedoids_visual_PCA(norm_cand, norm_y_pre, kmedoids, name='GF optimize: Iter {}'.format(GF_Iter))

center_cand = candidates[kmedoids.medoid_indices_]
center_y_pre = y_pre_list[kmedoids.medoid_indices_]

norm_center_cand = norm_cand[kmedoids.medoid_indices_]
norm_center_r_pre = cost_model(norm_center_cand).mean
center_r_pre = unnormalize(norm_center_r_pre, y_norm_bd_R)

data = torch.cat([obj_X, torch.tensor(center_cand)], dim=0)
pre_R = torch.cat([pre_R, center_r_pre.squeeze(-1)])
pre_GF = torch.cat([pre_GF, torch.tensor(center_y_pre.squeeze(-1))])

data = torch.cat([data, pre_R[:, None]], dim=1)
data = torch.cat([data, pre_GF[:, None]], dim=1)
fpath = './data/step3_GFopt/GF_{}.csv'.format(GF_Iter)
save_csv(data, fpath, name='GFopt')

print('Infeasible_solution_num={}'.format(decision_maker.infeasible_solution_num))
