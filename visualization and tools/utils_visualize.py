from scipy.spatial import KDTree
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *


def plt_train_R_min(fpath):
    R_Iter = 20
    next_num = 10
    # get observations
    X, R, label, pre_R = get_data(fpath, name='Ropt')
    best_X_list = []
    best_R_list = []
    x_list = []
    for i in range(R_Iter):
        l_bd = i * next_num
        r_bd = (i + 1) * next_num
        slice_X = X[l_bd:r_bd, :]
        slice_R = R[l_bd:r_bd]
        best_R, idx = slice_R.min(dim=0)
        best_X = slice_X[idx]

        x_list.append(i)
        best_X_list.append(best_X.numpy())
        best_R_list.append(best_R)

    plt.figure(figsize=(8, 6))
    x_tick = np.arange(0, R_Iter, 1)
    plt.xticks(x_tick)
    plt.xlim((-0.5, R_Iter - 0.5))

    plt.plot(x_list, best_R_list)
    for i in x_list:
        plt.annotate(best_X_list[i], (i, best_R_list[i]))

    plt.scatter(x_list, best_R_list, c='#FF0000', marker='*', edgecolor='none', alpha=None, s=80)
    plt.xlabel('Iter')
    plt.ylabel('R min')
    plt.title('R opt with multiscale BO')

    plt.show()


def cost_model_visualize(fpath):
    random_seed = 1
    np.random.seed(random_seed)

    # get R dataset
    cost_X, R, label, _ = get_data(fpath, name='Ropt')
    cost_x = normalize(cost_X, x_norm_bd)
    cost_y = normalize(R, y_norm_bd_R)

    cost_model, cost_likelihood = fit_gp_model(cost_x, cost_y, name='Cost_model', num_train_iters=500)
    print_model_info(cost_model)

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

    n = 5
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    x3 = np.linspace(0, 1, n)
    x4 = np.linspace(0, 1, n)
    norm_x = np.array(list(itertools.product(x1, x2, x3, x4)))
    norm_x_2d = reducer_loaded.transform(norm_x)
    norm_y = cost_model(torch.tensor(norm_x)).mean
    l = norm_y > 0.2

    ax.scatter(norm_x_2d[:, 0], norm_x_2d[:, 1], c=l, cmap='viridis', edgecolor='none', alpha=0.8, s=10)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_title("Cost_model visualization")
    plt.show()


def cost_model_visualize_heatmap(fpath):
    random_seed = 1
    np.random.seed(random_seed)

    # get R dataset
    cost_X, R, label, _ = get_data(fpath, name='Ropt')
    cost_x = normalize(cost_X, x_norm_bd)
    cost_y = normalize(R, y_norm_bd_R)

    cost_model, cost_likelihood = fit_gp_model(cost_x, cost_y, name='Cost_model', num_train_iters=500)
    print_model_info(cost_model)

    x1 = np.linspace(0, 1, 55)
    x2 = np.linspace(0, 1, 51)
    x3 = np.linspace(0, 1, 8)
    x4 = np.linspace(0, 1, 17)

    hypercube = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)
    reducer_loaded = joblib.load("umap_model.pkl")
    hypercube_2d = reducer_loaded.transform(hypercube)

    hull = ConvexHull(hypercube_2d)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 22

    fig, ax = plt.subplots(figsize=(8, 6))
    for simplex in hull.simplices:
        ax.plot(hypercube_2d[simplex, 0], hypercube_2d[simplex, 1], 'r--', lw=2.5)

    n = 10
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    x3 = np.linspace(0, 1, n)
    x4 = np.linspace(0, 1, n)
    norm_x = np.array(list(itertools.product(x1, x2, x3, x4)))
    norm_x_2d = reducer_loaded.transform(norm_x)
    norm_y = cost_model(torch.tensor(norm_x)).mean
    l = norm_y < 0.2

    grid_size = 100
    x_min, x_max = hypercube_2d[:, 0].min(), hypercube_2d[:, 0].max()
    y_min, y_max = hypercube_2d[:, 1].min(), hypercube_2d[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    kdtree = KDTree(norm_x_2d)
    distances, indices = kdtree.query(grid_points, k=10)
    grid_labels = l[indices].numpy().astype(int)

    prob_yes = np.mean(grid_labels, axis=1)
    prob_yes = prob_yes.reshape(grid_size, grid_size)
    im = ax.imshow(prob_yes, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="coolwarm", alpha=0.6)

    # cbar = fig.colorbar(im, ax=ax, label="Probability of 'Yes' Label")
    cbar = fig.colorbar(im, ax=ax)

    ax.set_xlabel("UMAP Dimension 1", fontsize=28, fontname='Arial')
    ax.set_ylabel("UMAP Dimension 2", fontsize=28, fontname='Arial')
    # ax.set_title("Cost_model visualization")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.show()
