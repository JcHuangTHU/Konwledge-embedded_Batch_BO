"""
  This program is based on the Knowledge-embedded batch BO to recommend next batch of experimental points,
  in order to find the laser conditions corresponding to the lowest line resistance

  Parameters:
  ------
  R_Iter:
    Denotes the R_Iter iteration in the resistance optimization,
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
from sklearn_extra.cluster import KMedoids
from gpytorch.kernels import MaternKernel, RBFKernel

R_Iter = 19
next_num = 10
scale_num = 20
random_seed = 1
np.random.seed(random_seed)

# get observations
X, R, label, pre_R = get_data('./data/step2_Ropt/Ropt_iter{}.csv'.format(R_Iter - 1), name='Ropt')
train_x = normalize(X, x_norm_bd)
train_y = normalize(R, y_norm_bd_R)
best_y = train_y.min()

decision_maker = Multiscale_BO_decision_maker(train_x, train_y, best_y)
decision_maker.get_search_batch()
decision_maker.get_infeasible_domain(label)

cand_list = torch.empty(0, 4)
y_pre_list = torch.empty(0, 1)
kernels = [MaternKernel(nu=0.5), MaternKernel(nu=1.5), MaternKernel(nu=2.5), RBFKernel()]
length_scale_array = 10 ** (np.linspace(-3, 3, scale_num))
np.random.shuffle(length_scale_array)

for kernel in kernels:
    decision_maker.register_model(kernel)
    decision_maker.model.mean_module.constant.data.fill_(0.5)  # Constant prior
    for i in length_scale_array:
        kernel_i = kernel
        kernel_i.lengthscale = i
        decision_maker.model_step(kernel_i)

        cand, y_pre = decision_maker.recommend(policy='ei', maximize=False)

        if cand is not None:
            cand_list = torch.cat([cand_list, cand[None, :]], dim=0)
            y_pre_list = torch.cat([y_pre_list, y_pre[None, :]], dim=0)

cand_list = np.array(cand_list.detach().numpy())
y_pre_list = np.array(y_pre_list.detach().numpy())
candidates, idx = np.unique(cand_list, axis=0, return_index=True)
y_pre_list = y_pre_list[idx]
norm_cand = normalize(torch.tensor(candidates), x_norm_bd)
norm_y_pre = normalize(torch.tensor(y_pre_list), y_norm_bd_R)

kmedoids = KMedoids(n_clusters=next_num,
                    method='pam',
                    init='k-medoids++',
                    max_iter=500,
                    random_state=1).fit(norm_cand)

kmedoids_visual(norm_cand, norm_y_pre, kmedoids, name='R optimize: Iter {}'.format(R_Iter))
kmedoids_visual_PCA(norm_cand, norm_y_pre, kmedoids, name='R optimize: Iter {}'.format(R_Iter))

center_cand = candidates[kmedoids.medoid_indices_]
center_y_pre = y_pre_list[kmedoids.medoid_indices_]

data = torch.cat([X, torch.tensor(center_cand)], dim=0)
pre_R = torch.cat([pre_R, torch.tensor(center_y_pre.squeeze(-1))])
data = torch.cat([data, pre_R[:, None]], dim=1)
fpath = './data/step2_Ropt/Ropt_iter{}.csv'.format(R_Iter)
save_csv(data, fpath, name='Ropt')

print('Infeasible_solution_num={}'.format(decision_maker.infeasible_solution_num))
