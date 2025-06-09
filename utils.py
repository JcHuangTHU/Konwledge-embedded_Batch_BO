import pandas as pd
import numpy as np
import torch
import gpytorch
import botorch
from botorch.utils.transforms import unnormalize, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import itertools
import joblib
from scipy.spatial import ConvexHull

bound = np.array([[3, 30], [0, 5], [150, 500], [20, 100]])
spacing = np.array([0.5, 0.1, 50, 5])
kwargs = {"dtype": torch.double, }
x_norm_bd = torch.tensor(bound.T)
# for R optimization
y_norm_bd_R = torch.tensor([[0], [500]])
# for average GF optimization
y_norm_bd_GF = torch.tensor([[0], [500]])


class ConstantMeanGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Multiscale_BO_decision_maker:

    def __init__(self, train_x, train_y, best_y):
        self.best_y = best_y
        self.raw_train_x = train_x
        self.raw_train_y = train_y
        self.train_x = train_x
        self.train_y = train_y

        self.search_batch = []
        self.infeasible_domain = torch.empty(0, 4, **kwargs)
        self.batch_size = 1e4  # Segmentation scale of search space
        self.infeasible_solution_num = 0

        self.model = None
        self.noise = 1e-4
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def get_search_batch(self):
        eps = 1e-6
        p = np.arange(bound[0, 0], bound[0, 1] + eps, spacing[0])
        df = np.arange(bound[1, 0], bound[1, 1] + eps, spacing[1])
        f = np.arange(bound[2, 0], bound[2, 1] + eps, spacing[2])
        v = np.arange(bound[3, 0], bound[3, 1] + eps, spacing[3])
        X_search = np.round(np.array(list(itertools.product(p, df, f, v))), decimals=3)
        print('Whole search space:', X_search.shape)
        X_search = torch.tensor(X_search)
        search_size = len(X_search)
        for i in range(math.ceil(search_size / self.batch_size)):
            idx_l = int(i * self.batch_size)
            idx_r = int(min((i + 1) * self.batch_size, search_size))
            self.search_batch.append(X_search[idx_l:idx_r])
        del X_search

    def get_infeasible_domain(self, label, X=None):
        if X is None:
            X = unnormalize(self.raw_train_x.clone(), x_norm_bd)

        eps = 1e-6
        round_tolerance = 3  # Floating-point representation causes error in determining numerical equality
        # Retaining decimals is equivalent to a given tolerance
        infea_dm = np.empty((0, 4), np.float64)
        for k in range(len(label)):
            if label[k] == 1:  # insufficient annealing
                p = np.arange(bound[0, 0], X[k, 0] + eps, spacing[0])
                df = np.array([X[k, 1]])
                f = np.array([X[k, 2]])
                v = np.arange(X[k, 3], bound[3, 1] + eps, spacing[3])
                product = np.array(list(itertools.product(p, df, f, v)))
                infea_dm = np.vstack((infea_dm, product))

            if label[k] == 2:  # excessive annealing
                p = np.arange(X[k, 0], bound[0, 1] + eps, spacing[0])
                df = np.array([X[k, 1]])
                f = np.array([X[k, 2]])
                v = np.arange(bound[3, 0], X[k, 3] + eps, spacing[3])
                product = np.array(list(itertools.product(p, df, f, v)))
                infea_dm = np.vstack((infea_dm, product))

        infea_dm = np.round(infea_dm, decimals=round_tolerance)
        infea_dm = np.unique(infea_dm, axis=0)
        print('Infeasible domain: ', infea_dm.shape[0])
        self.infeasible_domain = torch.tensor(infea_dm, **kwargs)

    def register_model(self, kernel):
        self.train_x = self.raw_train_x.clone()
        self.train_y = self.raw_train_y.clone()
        self.model = ConstantMeanGPModel(self.raw_train_x, self.raw_train_y, self.likelihood, kernel)
        self.model.likelihood.noise = self.noise
        self.model.eval()
        self.likelihood.eval()

    def model_step(self, kernel):
        self.model = ConstantMeanGPModel(self.train_x, self.train_y, self.likelihood, kernel)

    def recommend(self, policy, maximize, cost_model=None):
        if policy == 'ei':
            ei = botorch.acquisition.analytic.ExpectedImprovement(
                model=self.model,
                best_f=self.best_y,
                maximize=maximize)

            candidate, y_pre, norm_cand, norm_y_pre = self.get_candidate(policy=ei, y_norm_bd=y_norm_bd_R)

            if any((candidate == self.infeasible_domain).all(1)):
                self.infeasible_solution_num = self.infeasible_solution_num + 1
                self.train_x = torch.cat((self.train_x, norm_cand[None, :]), dim=0)
                self.train_y = torch.cat((self.train_y, torch.tensor([1])), dim=0)
                print('Infeasible solution appears: {}'.format(candidate))
                return None, None
            else:
                self.train_x = torch.cat((self.train_x, norm_cand[None, :]), dim=0)
                norm_y_pre = min([max([norm_y_pre, 0]), 1])
                norm_y_pre = torch.tensor([norm_y_pre])
                y_pre = unnormalize(norm_y_pre, y_norm_bd_R)
                self.train_y = torch.cat((self.train_y, norm_y_pre), dim=0)
                return candidate, y_pre

        if policy == 'cei':
            cei = botorch.acquisition.analytic.ConstrainedExpectedImprovement(
                model=botorch.models.model_list_gp_regression.ModelListGP(self.model, cost_model),
                best_f=self.best_y,
                objective_index=0,
                constraints={1: (None, 0.2)},
                maximize=maximize)

            candidate, y_pre, norm_cand, norm_y_pre = self.get_candidate(policy=cei, y_norm_bd=y_norm_bd_GF)

            if any((candidate == self.infeasible_domain).all(1)):
                self.infeasible_solution_num = self.infeasible_solution_num + 1
                self.train_x = torch.cat((self.train_x, norm_cand[None, :]), dim=0)
                self.train_y = torch.cat((self.train_y, torch.tensor([0])), dim=0)
                print('Infeasible solution appears: {}'.format(candidate))
                return None, None
            else:
                self.train_x = torch.cat((self.train_x, norm_cand[None, :]), dim=0)
                norm_y_pre = min([max([norm_y_pre, 0]), 1])
                norm_y_pre = torch.tensor([norm_y_pre])
                y_pre = unnormalize(norm_y_pre, y_norm_bd_GF)
                self.train_y = torch.cat((self.train_y, norm_y_pre), dim=0)
                return candidate, y_pre

    def get_candidate(self, policy, y_norm_bd):
        score_max = -torch.inf
        candidate = None
        for batch in self.search_batch:
            with torch.no_grad():
                norm_batch = normalize(batch, x_norm_bd)
                score = policy(norm_batch[:, None])
                slice_max, max_idx = torch.max(score, dim=0)
                if score_max < slice_max:
                    score_max = slice_max

                    norm_cand = norm_batch[max_idx]
                    candidate = batch[max_idx]
                    norm_y_pre = self.model(norm_cand[None, :]).mean
                    y_pre = unnormalize(norm_y_pre, y_norm_bd)

        print('candidate={}, y_pre={}, policy_score={}'.format(candidate.detach(), y_pre.detach(), score_max))
        return candidate.detach(), y_pre.detach(), norm_cand.detach(), norm_y_pre.detach()


def get_data(f_path, name):
    df = pd.read_csv(f_path, encoding='gbk')
    df.set_index('Number', inplace=True)
    X = torch.tensor(np.array(df.iloc[:, :4]), **kwargs)
    if name == 'Ropt':
        pre_R = torch.tensor(np.array(df.iloc[:, 4]), **kwargs)
        R = torch.tensor(np.array(df.iloc[:, 5]), **kwargs)
        label = torch.tensor(np.array(df.iloc[:, 6]))
        return X, R, label, pre_R

    if name == 'GFopt':
        pre_R = torch.tensor(np.array(df.iloc[:, 4]), **kwargs)
        pre_GF = torch.tensor(np.array(df.iloc[:, 5]), **kwargs)
        avg_GF = torch.tensor(np.array(df.iloc[:, 7]), **kwargs)
        return X, avg_GF, pre_R, pre_GF,


def save_csv(data, fpath, name):
    data = np.array(data.detach().numpy())
    df = None
    if name == 'Ropt':
        df = pd.DataFrame(data, index=np.arange(1, len(data) + 1, 1),
                          columns=['power (%)', 'df (mm)', 'frequency (kHz)', 'speed (mm/s)',
                                   'predict_R (ohm)'])

    if name == 'GFopt':
        df = pd.DataFrame(data, index=np.arange(1, len(data) + 1, 1),
                          columns=['power (%)', 'df (mm)', 'frequency (kHz)', 'speed (mm/s)',
                                   'predict_R (ohm)', 'predict_avg_GF'])

    df.index.name = 'Number'
    df.to_csv(fpath)
    print(df)


def kmedoids_visual(X, y, model, name):
    label = model.labels_
    x1 = np.linspace(0, 1, 55)
    x2 = np.linspace(0, 1, 51)
    x3 = np.linspace(0, 1, 8)
    x4 = np.linspace(0, 1, 17)

    hypercube = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)
    reducer_loaded = joblib.load("./visualization and tools/umap_model.pkl")
    hypercube_2d = reducer_loaded.transform(hypercube)

    hull = ConvexHull(hypercube_2d)
    fig, ax = plt.subplots(figsize=(8, 6))
    for simplex in hull.simplices:
        ax.plot(hypercube_2d[simplex, 0], hypercube_2d[simplex, 1], 'r--', lw=1.5, label="laser conditions 2d")

    X_2d = reducer_loaded.transform(X)
    idx = model.medoid_indices_
    ax.scatter(X_2d[:, 0], X_2d[:, 1], color='blue', label="multi-scale suggestions", s=20)
    ax.scatter(X_2d[idx, 0], X_2d[idx, 1], color='red', label="batch suggestions", s=80, marker='*')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_title(name)
    plt.show()


def kmedoids_visual_PCA(X, y, model, name):
    label = model.labels_
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, edgecolor='none', alpha=None, s=40)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(name)

    idx = model.medoid_indices_
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c='#FF0000', marker='*', alpha=None, s=80)
    plt.show()


def print_model_info(model):
    mean = model.mean_module.constant.item()
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()
    outputscale = model.covar_module.outputscale.detach().numpy()
    print('R GP model info: mean={}, lengthscale={}, outputscale={}\n'.format(mean, lengthscale, outputscale))


def visualize_train_gp(num_train_iters, losses, name):
    plt.figure()
    x = np.arange(1, num_train_iters + 1, 1)
    plt.plot(x, losses[:], c='r')
    plt.rcParams['font.family'] = 'Arial'
    plt.xlabel('Train iterations', fontsize=28)
    plt.ylabel('Negative marginal likelihood', fontsize=28)
    plt.title('{} trained by R-dataset'.format(name), fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=22, width=2)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')
    plt.show()


def fit_gp_model(train_x, train_y, num_train_iters, name):
    noise = 1e-4
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=4))
    model = ConstantMeanGPModel(train_x, train_y, likelihood, kernel)
    model.likelihood.noise = noise

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()
    losses = []
    for _ in tqdm(range(num_train_iters)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    visualize_train_gp(num_train_iters, losses, name)
    return model, likelihood
