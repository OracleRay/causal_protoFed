import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGRUCell(nn.Module):
    """
    步骤 S4: 基于图门控循环单元的时空预测
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GraphGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 图卷积的权重 (简化版GCN)
        self.W_u = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def normalize_adj(self, adj, epsilon=1e-6):
        """
        归一化邻接矩阵: D^{-1/2} * A * D^{-1/2}
        """
        # 计算度矩阵 [B, N]
        degree = adj.sum(dim=-1)
        # 避免除零
        degree_inv_sqrt = torch.pow(degree + epsilon, -0.5)
        # 构造对角矩阵 [B, N, N]
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        # 归一化
        adj_normalized = torch.bmm(torch.bmm(degree_inv_sqrt, adj), degree_inv_sqrt)
        return adj_normalized

    def forward(self, inputs, h_prev, adj):
        """
        inputs: [Batch, Nodes, Feat]
        h_prev: [Batch, Nodes, Hidden]
        adj: [Batch, Nodes, Nodes] (动态图)
        """
        # 归一化邻接矩阵
        adj_norm = self.normalize_adj(adj)

        # 拼接输入和上一时刻状态
        concat_input = torch.cat([inputs, h_prev], dim=-1)  # [B, N, F+H]

        # GRU 门控逻辑
        # 1. 更新门和重置门
        val_u = self.W_u(concat_input)
        val_r = self.W_r(concat_input)

        agg_u = torch.bmm(adj_norm, val_u)  # [B, N, H]
        agg_r = torch.bmm(adj_norm, val_r)

        u = torch.sigmoid(agg_u)  # 更新门
        r = torch.sigmoid(agg_r)  # 重置门

        # 2. 候选状态 (使用重置后的隐状态)
        c_input = torch.cat([inputs, r * h_prev], dim=-1)
        val_c = self.W_c(c_input)
        agg_c = torch.bmm(adj_norm, val_c)
        c = torch.tanh(agg_c)

        # 3. 更新隐状态
        h_new = u * h_prev + (1 - u) * c
        h_new = self.dropout(h_new)

        return h_new


class FeatureDecoupler(nn.Module):
    """
    步骤 S2: 客户端因果特征解耦
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeatureDecoupler, self).__init__()

        # 因果特征编码器 (更深的网络)
        self.encoder_inv = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 伪相关特征编码器 (较浅的网络,捕获表面模式)
        self.encoder_spu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(self, x):
        """
        x shape: [Batch, Nodes, Input_Dim]
        """
        h_inv = self.encoder_inv(x)
        h_spu = self.encoder_spu(x)
        return h_inv, h_spu


class EnvironmentClassifier(nn.Module):
    """
    步骤 S2: 环境识别辅助分类器
    用于计算 L_aux，强制伪相关特征预测环境标签
    """

    def __init__(self, hidden_dim, num_env_classes):
        super(EnvironmentClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_env_classes)

    def forward(self, h_spu):
        return self.classifier(h_spu)


class CausalGraphGenerator(nn.Module):
    """
    步骤 S3 & 权利要求4: 构建因果时空动态图
    """

    def __init__(self, hidden_dim, d_model, top_k=10):
        super(CausalGraphGenerator, self).__init__()
        self.W1 = nn.Linear(hidden_dim, d_model)
        self.W2 = nn.Linear(hidden_dim, d_model)
        self.top_k = top_k  # 保留top_k个最强连接,增加稀疏性

    def forward(self, h_inv):
        """
        h_inv: [Batch, Nodes, Hidden]
        返回: [Batch, Nodes, Nodes] 有向因果图
        """
        # 映射到源节点和目标节点空间
        M1 = torch.tanh(self.W1(h_inv))  # [B, N, d]
        M2 = torch.tanh(self.W2(h_inv))  # [B, N, d]

        # 计算反对称差异: M1 * M2^T - M2 * M1^T
        term1 = torch.bmm(M1, M2.transpose(1, 2))  # [B, N, N]
        term2 = torch.bmm(M2, M1.transpose(1, 2))  # [B, N, N]

        diff = term1 - term2  # 反对称矩阵

        # 使用sigmoid而不是ReLU(tanh()),保持反对称性的同时归一化到[0,1]
        # 正值表示因果关系强度
        A_causal = torch.sigmoid(diff)

        # 可选: Top-K稀疏化
        if self.top_k is not None:
            A_causal = self.top_k_sparsify(A_causal, self.top_k)

        # 添加自连接
        batch_size, num_nodes = A_causal.size(0), A_causal.size(1)
        identity = torch.eye(num_nodes, device=A_causal.device).unsqueeze(0).expand(batch_size, -1, -1)
        A_causal = A_causal + identity

        return A_causal

    def top_k_sparsify(self, adj, k):
        """
        保留每个节点的top-k个最强入边,其余置零
        """
        batch_size, num_nodes, _ = adj.size()

        # 对每一行(入边)找top-k
        topk_values, topk_indices = torch.topk(adj, k=min(k, num_nodes), dim=-1)

        # 创建mask
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_indices, 1)

        return adj * mask


class LocalTrafficModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_env_classes, graph_embed_dim=32):
        super(LocalTrafficModel, self).__init__()

        # 1. 解耦
        self.decoupler = FeatureDecoupler(input_dim, hidden_dim)

        # 2. 环境分类器
        self.env_classifier = EnvironmentClassifier(hidden_dim, num_env_classes)

        # 3. 动态图生成
        self.graph_gen = CausalGraphGenerator(hidden_dim, graph_embed_dim)

        # 4. 预测网络 (GraphGRU)
        self.gru = GraphGRUCell(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, output_dim)  # 预测未来流量

        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev=None):
        """
        前向传播
        x: [Batch, Nodes, Input_Dim]
        """
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), x.size(1), self.hidden_dim).to(x.device)

        # S2: 特征解耦
        h_inv, h_spu = self.decoupler(x)

        # 辅助任务: 环境预测 (用于计算 L_aux)
        env_logits = self.env_classifier(h_spu)

        # S3: 构建动态图
        adj = self.graph_gen(h_inv)

        # S4: 时空预测聚合
        h_new = self.gru(h_inv, h_prev, adj)

        # 预测输出
        prediction = self.predictor(h_new)

        return prediction, env_logits, h_inv, h_new