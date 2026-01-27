import torch
import torch.nn.functional as F


class LossCalculator:
    """
    包含专利中提到的四种损失函数
    """

    def __init__(self, lambda1=0.01, lambda2=0.001, lambda3=0.01, temp=0.5):
        self.lambda1 = lambda1  # 因果不变性损失权重
        self.lambda2 = lambda2  # 环境辅助损失权重
        self.lambda3 = lambda3  # 跨域对比损失权重
        self.temp = temp  # 对比学习温度系数

    def calc_pred_loss(self, y_pred, y_true):
        # 预测误差损失 (MSE)
        return F.mse_loss(y_pred, y_true)

    def calc_aux_loss(self, env_logits, env_labels):
        # 步骤 S2: 环境识别辅助损失 (Cross Entropy)
        # env_labels: [Batch]
        # 这里需要将 logits 展平或者对每个节点做分类，假设是对整图环境或每个节点环境
        # 假设所有节点共享同一个环境标签
        B, N, C = env_logits.shape
        env_logits_pool = torch.mean(env_logits, dim=1)  # Average pooling over nodes
        return F.cross_entropy(env_logits_pool, env_labels)

    def calc_inv_loss(self, h_inv, env_labels):
        # 步骤 S2 & 权利要求3: 因果不变性损失 (MMD变体)
        # 约束因果特征在不同环境标签下的分布差异最小化

        unique_labels = torch.unique(env_labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0).to(h_inv.device)

        loss_inv = 0.0
        # 简单实现：计算不同环境组之间的均值差异（Mean Discrepancy）
        # 专利公式：|| Mean(A) - Mean(B) ||^2

        # 将Batch分为两组（简化处理，实际可能需要两两组合）
        mask_a = (env_labels == unique_labels[0])
        mask_b = (env_labels != unique_labels[0])

        if mask_a.sum() == 0 or mask_b.sum() == 0:
            return torch.tensor(0.0).to(h_inv.device)

        feat_a = h_inv[mask_a].mean(dim=0)  # [Nodes, Hidden]
        feat_b = h_inv[mask_b].mean(dim=0)  # [Nodes, Hidden]

        loss_inv = torch.norm(feat_a - feat_b, p=2) ** 2
        return loss_inv

    def calc_contrast_loss(self, local_embeddings, global_prototypes):
        """
        S6 & 权利要求8: 跨域原型对比损失
        """
        if global_prototypes is None:
            return torch.tensor(0.0).to(local_embeddings.device)

        # [关键修正 1]: 切断对全局原型的梯度流
        # 我们只更新本地 Encoder 让其产生的特征 Z 去靠近全局原型 P，
        # 全局原型 P 的更新应仅由服务器聚合(S7)完成，不参与本地反向传播。
        prototypes = global_prototypes.detach()

        # 归一化，准备计算余弦相似度
        z_norm = F.normalize(local_embeddings, dim=1)  # [N, d]
        p_norm = F.normalize(prototypes, dim=1)        # [M, d]

        # [关键修正 2]: 计算 Logits
        # logits: [Batch*Nodes, Num_Prototypes]
        logits = torch.matmul(z_norm, p_norm.T) / self.temp

        # [关键修正 3]: 确定正样本 (Hard Assignment)
        # 对于每个本地特征，在当前的全局原型库中找到最相似的那一个作为"正样本"(Positive Target)
        # 其他所有原型自动作为"负样本"(Negatives)
        # values, target_indices: [Batch*Nodes]
        _, target_indices = torch.max(logits, dim=1)

        # [关键修正 4]: 计算 InfoNCE Loss
        # 公式: L = -log( exp(sim_pos / temp) / sum(exp(sim_all / temp)) )
        # PyTorch 的 CrossEntropyLoss 已经在内部高效且数值稳定地实现了 LogSoftmax + NLLLoss
        loss_contrast = F.cross_entropy(logits, target_indices)

        return loss_contrast