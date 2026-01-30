import torch
import numpy as np
import random
import torch.nn.functional as F
from utils.loss_cal import LossCalculator as lossCalculator
from sklearn.cluster import KMeans
from model.st_model import LocalTrafficModel as Model


class FederatedClient:
    def __init__(self, client_id, model_config):
        self.client_id = client_id
        self.model = Model(**model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_calc = lossCalculator()
        self.local_prototypes = None  # Step S5 生成
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_env_classes = int(model_config.get('num_env_classes', 4))

    def train_epoch(self, dataloader, global_prototypes, num_batches=32):
        """
        本地训练循环
        global_prototypes: 服务器下发的全局状态原型库 (P_old)
        num_batches: 随机采样的batch数量，默认32
        """
        self.model.train()
        loss = 0
        all_embeddings = []  # 用于S5聚类

        if global_prototypes is not None:
            global_prototypes = global_prototypes.to(self.device)

        # ====== 新增：随机采样batch索引 ======
        total_batches = len(dataloader)
        if total_batches <= num_batches:
            selected_indices = set(range(total_batches))
        else:
            selected_indices = set(random.sample(range(total_batches), num_batches))
        # ====================================

        count = 0
        all_pred_losses = 0
        for batch_idx, batch in enumerate(dataloader):
            # ====== 新增：跳过未被选中的batch ======
            if batch_idx not in selected_indices:
                continue
            # ====================================

            # batch: (x, y, adj_matrix)
            x, y, _ = batch
            # x: [B, N, his_num], y: [B, N, pred_num]
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            B = x.size(0)
            env_labels = torch.randint(0, self.num_env_classes, (B,), device=self.device)

            self.optimizer.zero_grad()

            # 前向传播
            y_pred, env_logits, h_inv, h_final = self.model(x)

            # 收集嵌入用于生成本地原型 (Step S5)
            # h_final作为节点嵌入 Z_t
            all_embeddings.append(h_final.detach().cpu().numpy())

            # 计算各项损失
            # 1. 预测损失
            l_pred = self.loss_calc.calc_pred_loss(y_pred, y)
            l_pred.backward()
            self.optimizer.step()

            count += 1
            all_pred_losses += l_pred
            if count == num_batches:
                # 2. 辅助分类损失
                l_aux = self.loss_calc.calc_aux_loss(env_logits, env_labels)

                # 3. 因果不变性损失
                l_inv = self.loss_calc.calc_inv_loss(h_inv, env_labels)

                # 4. 对比损失 (需要flatten节点维度)
                # [B, N, H] -> [B*N, H]
                flat_h = h_final.reshape(-1, h_final.size(-1))
                l_contrast = self.loss_calc.calc_contrast_loss(flat_h, global_prototypes)

                # 总损失 (步骤 S6)
                loss = all_pred_losses + \
                       self.loss_calc.lambda1 * l_inv + \
                       self.loss_calc.lambda2 * l_aux + \
                       self.loss_calc.lambda3 * l_contrast

        # 步骤 S5: 本地交通原型提取
        self.generate_local_prototypes(all_embeddings)

        return self.model.state_dict(), self.local_prototypes, loss

    def generate_local_prototypes(self, embeddings_list, k=5):
        """
        K-Means聚类生成本地原型
        """
        # 合并所有batch的嵌入: [Total_Samples * N, Hidden]
        data = np.concatenate([e.reshape(-1, e.shape[-1]) for e in embeddings_list], axis=0)

        # 为节省时间，可以下采样
        if data.shape[0] > 10000:
            indices = np.random.choice(data.shape[0], 10000, replace=False)
            data = data[indices]

        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(data)

        # 保存中心向量作为本地原型集合 P(k)
        self.local_prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

    def update_weights(self, global_weights, client_weights, eta=0.5):
        updated_weights = {}

        for key in client_weights.keys():
            # 对每个参数分别计算余弦相似度
            global_flat = global_weights[key].flatten()
            client_flat = client_weights[key].flatten()

            sim = F.cosine_similarity(
                global_flat.unsqueeze(0),
                client_flat.unsqueeze(0),
                dim=1
            ).item()

            # 如果相似则更新,否则保持原值
            if sim > 0:
                updated_weights[key] = (
                        client_weights[key] +
                        eta * sim * (global_weights[key] - client_weights[key])
                )
            else:
                updated_weights[key] = client_weights[key]

        return updated_weights