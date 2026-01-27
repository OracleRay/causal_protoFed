import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def asym_adj(adj):
    """非对称归一化邻接矩阵。"""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(city_name):
    """
    加载邻接矩阵，并进行非对称归一化
    :param city_name:
    :return:
    """
    path = "./dataset/" + city_name + "/matrix.npy"
    matrix = np.load(path)
    return [asym_adj(matrix), asym_adj(np.transpose(matrix))]


def load_dataset(city_name):
    """
    加载数据集，并标准化
    :param city_name:
    :return:
    """
    path = "./dataset/" + city_name + "/dataset.npy"
    dataset = np.load(path)
    print("load:", city_name, ",Dataset Shape:", dataset.shape)

    # dataset.npy: [Time, Nodes, Features] -> [Nodes, Features, Time]
    dataset = dataset.transpose((1, 2, 0)).astype(np.float32)

    # 标准化（按 feature 维，在 node+time 维度上做）
    dataset_means = np.mean(dataset, axis=(0, 2))  # [F]
    dataset_stds = np.std(dataset, axis=(0, 2))    # [F]
    dataset = dataset - dataset_means.reshape(1, -1, 1)
    dataset = dataset / (dataset_stds.reshape(1, -1, 1) + 1e-6)

    return dataset, dataset_means, dataset_stds


def split_dataset(options, dataset):
    """
    划分数据集为 x 和 y
    :param options:
    :param dataset:
    :return:
    """
    # dataset: [Nodes, Time]（单一特征）
    # 遍历时间轴，生成每个样本的起始和结束索引;
    indices = [(i, i + (options['his_num'] + options['pred_num'])) for i
               in range(dataset.shape[1] - (
                options['his_num'] + options['pred_num']) + 1)]

    # 划分不同的时间步为输入和输出
    features, target = [], []
    for i, j in indices:
        # 输入: [Nodes, his_num]
        features.append(dataset[:, i: i + options['his_num']])
        # 输出: [Nodes, pred_num]
        target.append(dataset[:, i + options['his_num']: j])

    # [Samples, Nodes, his_num/pred_num]
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))


def generate_dataset(options, city_name):
    """
    生成目标城市的训练集，验证集，测试集
    :param options:
    :param city_name:
    :return:
    """
    dataset, dataset_means, dataset_stds = load_dataset(city_name)
    adj_matrix = load_adj(city_name)
    adj_matrix = [torch.from_numpy(adj) for adj in adj_matrix]

    # 仅选用单一目标特征（默认 feature 0），把 [Nodes, Features, Time] -> [Nodes, Time]
    feature_idx = int(options.get('feature_idx', 0))
    dataset = dataset[:, feature_idx, :]

    # 生成训练集，验证集，测试集
    if city_name == options['target_city']:
        # 目标城市（服务器城市）划分：训练集（前7天），测试集（后20%）
        dataset_train = dataset[:, :288 * 7]
        dataset_test = dataset[:, int(dataset.shape[1] * 0.8):]

        x_train, y_train = split_dataset(options, dataset_train)
        x_test, y_test = split_dataset(options, dataset_test)
    else:
        x_inputs, y_outputs = split_dataset(options, dataset)
        x_train, x_test, y_train, y_test = train_test_split(x_inputs, y_outputs, test_size=0.2,
                                                            random_state=options['seed'])

    # 传递标准化参数到数据集
    train_dataset = TrafficDataset(x_train, y_train, adj_matrix, dataset_means, dataset_stds, feature_idx)
    test_dataset = TrafficDataset(x_test, y_test, adj_matrix, dataset_means, dataset_stds, feature_idx)
    return train_dataset, test_dataset


def compute_graph_stats(adj_matrix):
    """
    计算本地（当前客户端/城市）邻接矩阵的空间统计特征
    :param adj_matrix: 本地邻接矩阵
    :return:
    """
    G = nx.from_numpy_array(adj_matrix)  # 把邻接矩阵转为 networkx 的图对象
    degrees = [d for n, d in G.degree()]  # 计算图中每个节点的度
    # avg_degree = np.mean(degrees)  # 计算所有节点度的平均值
    degree_var = np.var(degrees)  # 计算所有节点度的方差
    avg_clustering = nx.average_clustering(G)  # 计算图的平均聚类系数
    return np.array([degree_var, avg_clustering])


class TrafficDataset(Dataset):
    def __init__(self, x_data, y_data, adj_matrix, dataset_means=None, dataset_stds=None, feature_idx=0):
        super().__init__()
        self.x_data = x_data  # [Samples, Nodes, his_num]
        self.y_data = y_data  # [Samples, Nodes, pred_num]
        self.adj_matrix = adj_matrix  # 邻接矩阵（当前模型未显式使用）
        self.dataset_means = dataset_means  # [F]
        self.dataset_stds = dataset_stds    # [F]
        self.feature_idx = int(feature_idx)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y, self.adj_matrix

    def __len__(self):
        return len(self.x_data)
