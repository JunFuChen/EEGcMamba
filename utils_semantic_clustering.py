import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim

class GaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, device):
        super(GaussianMixture, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.weights = nn.Parameter(torch.ones(n_components).to(device) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features).to(device))
        self.covs = nn.Parameter(torch.stack([torch.eye(n_features).to(device)] * n_components))

    def forward(self, x):
        mixture_dists = dist.MixtureSameFamily(
            mixture_distribution=dist.Categorical(probs=self.weights),
            component_distribution=dist.MultivariateNormal(loc=self.means, covariance_matrix=self.covs)
        )
        return mixture_dists.log_prob(x)

def train_gmm(X, n_components,device, n_epochs=10, lr=0.01):
    n_features = X.shape[1]
    X = X.to(device)
    gmm = GaussianMixture(n_components, n_features, device)
    print(list(gmm.parameters()))
    optimizer = optim.Adam(gmm.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        log_prob = gmm(X)
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return gmm

def get_cluster_labels_and_centers(gmm, X):
    component_dists = dist.MultivariateNormal(loc=gmm.means, covariance_matrix=gmm.covs)
    component_log_probs = component_dists.log_prob(X.unsqueeze(1))  # [batch_size, n_components]
    component_probs = torch.exp(component_log_probs)
    cluster_labels = torch.argmax(component_probs, dim=1)

    cluster_centers = gmm.means

    return cluster_labels, cluster_centers

#  test
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# X = torch.randn(100, 2).to(device)
# n_components = 3
# gmm = train_gmm(X, n_components, device=device)

# cluster_labels, cluster_centers = get_cluster_labels_and_centers(gmm, X)
# print("Cluster labels:")
# print(cluster_labels)
# print("Cluster centers (means):")
# print(cluster_centers)

###########################################################

def mean_shift(X, bandwidth,device, max_iter=100, tol=1e-3):
    """
    :param X: PyTorch Tensor，(n_samples, n_features)
    """
    n_samples, n_features = X.size()
    device = X.device

    # 初始化每个点的聚类标签为 -1（未分配）
    cluster_labels = torch.full((n_samples,), 99, dtype=torch.long, device=device)
    cluster_centers = []

    for i in range(n_samples):
        if cluster_labels[i] != 99:
            continue

        center = X[i].clone()
        for _ in range(max_iter):
            distances = torch.norm(X - center, dim=1)
            neighbors = X[distances <= bandwidth]
            new_center = torch.mean(neighbors, dim=0)
            if torch.norm(new_center - center) < tol:
                break
            center = new_center

        # 分配聚类标签
        distances = torch.norm(X - center, dim=1)
        cluster_mask = distances <= bandwidth
        cluster_labels[cluster_mask] = len(cluster_centers)
        cluster_centers.append(center)

    cluster_centers = torch.stack(cluster_centers)
    # need refined for nll-loss
    unique_labels, refined_labels = torch.unique(cluster_labels, return_inverse=True)

    return refined_labels, cluster_centers

# test mean-shift
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# X = torch.randn(100, 2).to(device)  # 生成一些随机数据
# bandwidth = 0.5
# cluster_labels, cluster_centers = mean_shift(X, bandwidth)
# print(cluster_labels)
# print(cluster_centers)


# spectral clustering

def KMeans(x, device,K=10, Niters=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)
        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)

        labels_count_all = torch.ones([K]).long().to(device)
        labels_count_all[unique_labels[:, 0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).to(device).scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c

def embeddings_to_cosine_similarity(E, sigma=1.0):
    dot = torch.abs_(torch.mm(E, E.t()))
    norm = torch.norm(E, 2, 1)
    x = torch.div(dot, norm)
    x = torch.div(x, torch.unsqueeze(norm, 0))
    x = x.div_(sigma)

    return torch.max(x, x.t()).fill_diagonal_(0)


def kway_normcuts(F, K=2, sigma=1.0):
    W = embeddings_to_cosine_similarity(F, sigma=sigma)
    degree = torch.sum(W, dim=0)
    D_pow = torch.diag(degree.pow(-0.5))
    L = torch.matmul(torch.matmul(D_pow, torch.diag(degree)-W), D_pow)
    _, eigenvector = torch.linalg.eigh(L, eigenvectors=True)
    eigvec_norm = torch.matmul(torch.diag(degree.pow(-0.5)), eigenvector)
    eigvec_norm = eigvec_norm/eigvec_norm[0][0]
    k_eigvec = eigvec_norm[:,:K]

    return k_eigvec


def spectral_clustering(F,device, K=10, clusters=10, Niters=10, sigma=1):
    k_eigvec = kway_normcuts(F, K=K, sigma=sigma)
    cl, _ = KMeans(k_eigvec, K=clusters, Niters=Niters, verbose=False)
    Ncl = cl.view(cl.size(0), 1).expand(-1, F.size(1))
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
    labels_count_all = torch.ones([clusters]).long().to(device)
    labels_count_all[unique_labels[:,0]] = labels_count
    c = torch.zeros([clusters, F.size(1)], dtype=torch.float).to(device).scatter_add_(0, Ncl, F)
    c = c / labels_count_all.float().unsqueeze(1)

    return cl, c


