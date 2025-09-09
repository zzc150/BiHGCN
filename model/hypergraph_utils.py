import torch
import numpy as np
import numpy as np
from functools import partial
from math import radians, cos, sin, asin, sqrt
from collections import Counter


# ========================================
# Hypergraph Related Functions from HGM2R
def dist2H(dist: torch.Tensor, top_k):
    H = torch.zeros_like(dist).long()
    _, tk_idx = dist.topk(top_k, dim=1, largest=False)
    col_idx = torch.arange(tk_idx.size(0)).unsqueeze(1).repeat(1, top_k)
    row_idx, col_idx = tk_idx.view(-1), col_idx.view(-1)
    H[row_idx, col_idx] = 1
    return H


def ft2H(ft: torch.Tensor, top_k):
    d = torch.cdist(ft, ft)
    if isinstance(top_k, list):
        Hs = []
        for _k in top_k:
            Hs.append(dist2H(d, _k))
        return Hs
    else:
        return dist2H(d, top_k)

#构造超图
def ft2G(ft: torch.Tensor, top_k=50, sym=False):
    Hs = None
    H = ft2H(ft, top_k)
    if isinstance(top_k, list):
        H = torch.stack(H)
    norm_r = 1 / H.sum(dim=1, keepdim=True)
    norm_r[torch.isinf(norm_r)] = 0
    norm_c = 1 / H.sum(dim=0, keepdim=True)
    norm_c[torch.isinf(norm_c)] = 0
    G = torch.matmul((norm_r * H), (norm_c * H).T)
    return G


#构造图
def gcn_ft2knn(fts, top_k=50):
    n = fts.size(0)
    A = torch.zeros((n, n))
    cdist = torch.cdist(fts, fts)
    _, tk_idx = cdist.topk(top_k, dim=1, largest=False)
    node_idx = torch.arange(tk_idx.size(0)).unsqueeze(1).repeat(1, top_k)
    A[tk_idx, node_idx] = 1
    A[node_idx, tk_idx] = 1
    norm_r = 1 / A.sum(dim=1, keepdim=True)
    norm_r[torch.isinf(norm_r)] = 0
    A = A * norm_r + torch.eye(n)
    return A.cuda()

# ========================================
#from sthgcn-icdm
# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = x.detach().cpu().numpy()
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def compute_dtw_matrix(data):
    N, T = data.shape
    dtw_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):  # Only compute upper triangle (including diagonal)
            dist= accelerated_dtw(data[i], data[j], 'euclidean')
            dtw_matrix[i, j] = dist
            dtw_matrix[j, i] = dist  # DTW distance is symmetric

    return dtw_matrix

def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    # print('DE',DE.shape)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    # print(invDE)
    invDE[np.isinf(invDE)]=0.0
    # print('invDE', invDE.shape)
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)]=0.0
    W = np.mat(np.diag(W))
    # print(W)
    H = np.mat(H)
    HT = H.T
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        # print(G.shape)
        # exit()
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(X.shape[2], -1)

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


# ========================================
#form DHGNN
"""
transform graphs (represented by edge list) to hypergraph (represented by node_dict & edge_dict)
"""
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances as cos_dis, euclidean_distances
from sklearn.cluster import KMeans
# from utils.layer_utils import sample_ids
# ========================================
#layer_utils
import torch
from torch import nn
import pandas as pd


def cos_dis(X):
        """
        cosine distance
        :param X: (N, d)
        :return: (N, N)
        """
        X = nn.functional.normalize(X)
        XT = X.transpose(0, 1)
        return torch.matmul(X, XT)


def sample_ids(ids, k):
    """
    sample `k` indexes from ids, must sample the centroid node itself
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k - 1, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    sampled_ids.append(ids[-1])  # must sample the centroid node itself
    return sampled_ids


def sample_ids_v2(ids, k):
    """
    purely sample `k` indexes from ids
    :param ids: indexes sampled from
    :param k: number of samples
    :return: sampled indexes
    """
    df = pd.DataFrame(ids)
    sampled_ids = df.sample(k, replace=True).values
    sampled_ids = sampled_ids.flatten().tolist()
    return sampled_ids
# ========================================

def edge_to_hyperedge(edges):
    """
    transform edges to hyperedges
    For hyperedges constructed by existed graph edges, hyperedge_id = centroid_node_id
    :param edge_list: list of edges (numpy array)
    :return: node_dict: edges containing the node
    :return: edge_dict: nodes contained in the edge
    """
    edge_list = [list() for i in range(edges.max()+1)]
    # node_cited = set()
    # node_list = [list() for i in range(edges.max()+1)]
    for edge in edges:
        # edge[0]: paper cited; edge[1]: paper citing
        edge_list[edge[0]].append(edge[1])
        edge_list[edge[1]].append(edge[0])
        # node_cited.add(edge[1])
    # print(len(node_cited))
    node_list = edge_list
    return node_list, edge_list



# def construct_H_with_KNN(X, K_neigs=[10], is_probH=False, m_prob=1):
#     """
#     init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
#     :param X: N_object x feature_number
#     :param K_neigs: the number of neighbor expansion
#     :param is_probH: prob Vertex-Edge matrix or binary
#     :param m_prob: prob
#     :return: N_object x N_hyperedge
#     """
#     if len(X.shape) != 2:
#         X = X.reshape(-1, X.shape[-1])
#
#     if type(K_neigs) == int:
#         K_neigs = [K_neigs]
#
#     dis_mat = cos_dis(X)
#     H = None
#     for k_neig in K_neigs:
#         H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
#         H = hyperedge_concat(H, H_tmp)
#     return H


# def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
#     """
#     construct hypregraph incidence matrix from hypergraph node distance matrix
#     :param dis_mat: node distance matrix
#     :param k_neig: K nearest neighbor
#     :param is_probH: prob Vertex-Edge matrix or binary
#     :param m_prob: prob
#     :return: N_object X N_hyperedge
#     """
#     n_obj = dis_mat.shape[0]
#     # construct hyperedge from the central feature space of each node
#     n_edge = n_obj
#     H = np.zeros((n_obj, n_edge))
#     for center_idx in range(n_obj):
#         dis_mat[center_idx, center_idx] = 0
#         dis_vec = dis_mat[center_idx]
#         nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
#         avg_dis = np.average(dis_vec)
#         if not np.any(nearest_idx[:k_neig] == center_idx):
#             nearest_idx[k_neig - 1] = center_idx
#
#         for node_idx in nearest_idx[:k_neig]:
#             if is_probH:
#                 H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
#             else:
#                 H[node_idx, center_idx] = 1.0
#     return H


def _edge_dict_to_H(edge_dict):
    """
    calculate H from edge_list
    :param edge_dict: edge_list[i] = adjacent indices of index i
    :return: H, (n_nodes, n_nodes) numpy ndarray
    """
    n_nodes = len(edge_dict)
    H = np.zeros(shape=(n_nodes, n_nodes))
    for center_id, adj_list in enumerate(edge_dict):
        H[center_id, center_id] = 1.0
        for adj_id in adj_list:
            H[adj_id, center_id] = 1.0
    return H
def _edge_dict_to_H_with_weights(edge_dict):
    """
    Calculate H and W from edge_dict.
    :param edge_dict: edge_dict[i] = adjacent indices of index i (including possible repetitions)
    :return: H (n_nodes, n_nodes) and W (n_nodes, n_nodes), numpy ndarrays
    """
    n_nodes = len(edge_dict)
    H = np.zeros((n_nodes, n_nodes))  # 超图关联矩阵
    W = np.zeros((n_nodes, n_nodes))  # 权重矩阵

    for center_id, adj_list in enumerate(edge_dict):
        # 统计该超边中每个区域出现的次数
        count_dict = Counter(adj_list)
        H[center_id, center_id] = 1.0  # 自环连接
        W[center_id, center_id] = 1.0  # 自环的权重

        for adj_id, count in count_dict.items():
            H[adj_id, center_id] = 1.0
            W[adj_id, center_id] = count  # 根据重复次数设定权重

    return H, W

# def _generate_G_from_H(H, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     H = np.array(H)
#     n_edge = H.shape[1]
#     # the weight of the hyperedge
#     W = np.ones(n_edge)
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)
#
#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     W = np.mat(np.diag(W))
#     H = np.mat(H)
#     HT = H.T
#
#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         return G
#
#
# def generate_G_from_H(H, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     if type(H) != list:
#         return _generate_G_from_H(H, variable_weight)
#     else:
#         G = []
#         for sub_H in H:
#             G.append(generate_G_from_H(sub_H, variable_weight))
#         return G


def construct_G_from_fts(Xs, k_neighbors):
    """
    generate G from concatenated H from list of features
    :param Xs: list of features
    :param k_neighs: list of k
    :return: numpy array
    """
    Hs = [construct_H_with_KNN(Xs[i], [k_neighbors[i]]) for i in range(len(Xs))]
    H = np.concatenate(Hs, axis=1)
    G = generate_G_from_H(H)
    return G


def H_to_node_edge_dict(H):
    H = np.array(H, dtype=np.int)
    row, col = np.where(H==1)
    n_node, n_edge = H.shape[0], H.shape[1]
    node_dict = [list() for i in range(n_node)]
    edge_dict = [list() for i in range(n_edge)]
    for i in range(row.size):
        node_dict[row[i]].append(col[i])
        edge_dict[col[i]].append(row[i])
    return node_dict, edge_dict

def euclidean_dist(X):
    """
    Euclidean distance
    :param X: (N, d)
    :return: (N, N)
    """
    X_norm = (X ** 2).sum(1).view(-1, 1)
    dist = X_norm + X_norm.t() - 2.0 * torch.matmul(X, X.t())
    return torch.sqrt(torch.clamp(dist, min=0.0))
def _construct_edge_list_from_distance(X, k_neigh):
    """
    construct edge_list (numpy array) from kNN distance for single modality
    :param X -> numpy array: feature
    :param k_neigh -> int: # of neighbors
    :return: N * k_neigh numpy array
    """
    B, T , N,  = X.shape
    X = X.reshape(N, B* T)
    # dis = cos_dis(X)
    dis = euclidean_dist(X)
    # dis = compute_dtw_matrix(X)
    # dis = torch.Tensor(dis)
    _, k_idx = dis.topk(k_neigh, dim=-1, largest=False)
    k_idx = k_idx.cpu()
    return k_idx.numpy()


def construct_edge_list_from_knn(Xs, k_neighs):
    """
    construct concatenated edge list from list of features with kNN from multi-modal
    :param Xs: list of features
    :param k_neighs: list of k
    :return: concatenated edge list
    """
    L = [_construct_edge_list_from_distance(Xs[i], k_neighs[i]) for i in range(len(Xs))]
    # 检查 L 中的每个元素是否为 Tensor，并转换为 numpy.ndarray
    L = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in L]

    return np.concatenate(L, axis=1)

    # return np.concatenate([_construct_edge_list_from_distance(Xs[i], k_neighs[i]) for i in range(len(Xs))], axis=1)



def _construct_edge_list_from_cluster(X, clusters, adjacent_clusters, k_neighbors) -> np.array:
    """
    construct edge list (numpy array) from cluster for single modality
    :param X: feature
    :param clusters: number of clusters for k-means
    :param adjacent_clusters: a node's adjacent clusters
    :param k_neighbors: number of a node's neighbors
    :return:
    """
    N = X.shape[0]
    N, T, D = X.shape
    X = X.reshape(D * N, T)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    dis = euclidean_distances(X, centers)
    dis = dis.transpose(0, 1)
    cluster_center_dict = torch.topk(torch.Tensor(dis), adjacent_clusters, largest=False)
    # cluster_center_dict = cluster_center_dict.numpy()

    point_labels = kmeans.labels_
    point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(clusters)]

    def _list_cat(list_of_array):
        """
        example: [[0,1],[3,5,6],[-1]] -> [0,1,3,5,6,-1]
        :param list_of_array: list of np.array
        :return: list of numbers
        """
        ret = list()
        for array in list_of_array:
            ret += array.tolist()
        return ret

    cluster_neighbor_dict = [_list_cat([point_in_which_cluster[int(cluster_center_dict[point][i].item())]
                                        for i in range(adjacent_clusters)]) for point in range(N)]
    for point, entry in enumerate(cluster_neighbor_dict):
        entry.append(point)
    sampled_ids = [sample_ids(cluster_neighbor_dict[point], k_neighbors) for point in range(N)]
    return np.array(sampled_ids)

# def _construct_edge_list_from_cluster(X, clusters, adjacent_clusters, k_neighbors) -> np.array:
#     """
#     Construct edge list (numpy array) from cluster for single modality.
#
#     Parameters:
#     - X: Feature tensor of shape (T, N, D), where T is the number of time slices, N is the number of nodes,
#          and D is the number of features per node.
#     - clusters: Number of clusters for k-means.
#     - adjacent_clusters: Number of a node's adjacent clusters.
#     - k_neighbors: Number of a node's neighbors.
#
#     Returns:
#     - np.array: Concatenated edge list.
#     """
#     N, T, D = X.shape
#     X_flat = X.reshape(T * D, N)  # Reshape to (T * N, D) for k-means clustering across time slices
#     kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X_flat)
#     centers = kmeans.cluster_centers_
#     dis = euclidean_distances(X_flat, centers)
#     cluster_center_dict = torch.topk(torch.Tensor(dis), adjacent_clusters, largest=False)
#
#     # cluster_center_dict = cluster_center_dict.numpy()
#     point_labels = kmeans.labels_
#     point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(clusters)]
#
#     def _list_cat(list_of_array):
#         """
#         Concatenate lists of arrays.
#
#         Parameters:
#         - list_of_array: List of np.array.
#
#         Returns:
#         - list: Concatenated list of numbers.
#         """
#         ret = list()
#         for array in list_of_array:
#             ret += array.tolist()
#         return ret
#
#
#
#     cluster_neighbor_dict = [_list_cat([point_in_which_cluster[int(cluster_center_dict[point][i].item())]
#                                         for i in range(adjacent_clusters)]) for point in range(N)]
#     for point, entry in enumerate(cluster_neighbor_dict):
#         entry.append(point)
#     sampled_ids = [sample_ids(cluster_neighbor_dict[point], k_neighbors) for point in range(N)]
#     return np.array(sampled_ids)

def construct_edge_list_from_cluster(Xs, clusters, adjacent_clusters, k_neighbors) -> np.array:
    """
    construct concatenated edge list from list of features with cluster from multi-modal
    :param Xs: list of features of each modality
    :param clusters: list of number of clusters for k-means of each modality
    :param adjacent_clusters: list of number of a node's adjacent clusters of each modality
    :param k_neighbors: list of number of a node's neighbors
    :return: concatenated edge list (numpy array)
    """
    Xs = [xs.cpu().numpy() if torch.is_tensor(xs) else xs for xs in Xs]
    return np.concatenate([_construct_edge_list_from_cluster(Xs[i], clusters[i], adjacent_clusters[i], k_neighbors[i])
                           for i in range(len(Xs))], axis=1)

#----------
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf

def haversine(lon1, lat1, lon2, lat2): #

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1]


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
