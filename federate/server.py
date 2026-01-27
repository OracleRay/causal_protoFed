import torch
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 使用单线程避免内存泄露

from sklearn.cluster import KMeans


class FederatedServer:
    """
    步骤 S7: 云端双通道聚合
    """

    def __init__(self, global_model, k_global=10):
        self.global_model = global_model  # 初始化后的全局模型
        self.global_prototypes = None
        self.k_global = k_global

    def aggregate(self, client_weights, client_prototypes, client_loss):
        """
        1. 参数聚合 (基于损失)
        2. 原型聚合 (二次聚类)
        """
        # 1. 基于损失的加权聚合
        # 获取第一个客户端的权重作为模板
        first_client_key = next(iter(client_weights))
        first_client_weights = client_weights[first_client_key]
        global_weights = {key: torch.zeros_like(value) for key, value in first_client_weights.items()}

        loss_sum = sum(1 / loss for loss in client_loss.values())

        # 筛选损失权重大于一半的客户，防止其主导模型走向
        for k, v in client_loss.items():
            v = 1 / v
            if v / loss_sum > 0.5:
                other_loss = loss_sum - v
                client_loss[k] = other_loss
                loss_sum = other_loss * 2
                break

        # 按照权重累加客户端参数
        for client, client_weight in client_weights.items():
            alpha = (1 / client_loss[client]) / loss_sum
            for k, v in client_weight.items():
                global_weights[k] += v * alpha  # 根据权重累加私有参数

        # 更新全局模型参数
        self.global_model.load_state_dict(global_weights)

        # 2. 原型融合
        # 收集所有客户端原型构建候选池 P_pool
        p_pool = torch.cat(list(client_prototypes.values()), dim=0).cpu().numpy()

        # 二次聚类生成全局原型库 P_new
        kmeans = KMeans(n_clusters=self.k_global, n_init='auto')
        kmeans.fit(p_pool)
        self.global_prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        return self.global_model.state_dict(), self.global_prototypes