import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGRUCell(nn.Module):
    """
    步骤 S4: 基于图门控循环单元的时空预测
    """

    def __init__(self, input_dim, hidden_dim):
        super(GraphGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        # 图卷积的权重 (简化版GCN)
        self.W_u = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs, h_prev, adj):
        # inputs: [Batch, Nodes, Feat]
        # h_prev: [Batch, Nodes, Hidden]
        # adj: [Batch, Nodes, Nodes] (动态图)

        # 拼接输入和上一时刻状态
        concat_input = torch.cat([inputs, h_prev], dim=-1)  # [B, N, F+H]

        # 图卷积操作: A * X * W (这里拆分为 A * (XW))
        # 1. 线性变换
        val_u = self.W_u(concat_input)
        val_r = self.W_r(concat_input)
        val_c = self.W_c(concat_input)

        # 2. 邻居聚合 (bmn * bnf -> bmf)
        agg_u = torch.einsum('bnm,bmf->bnf', adj, val_u)
        agg_r = torch.einsum('bnm,bmf->bnf', adj, val_r)
        agg_c = torch.einsum('bnm,bmf->bnf', adj, val_c)

        # GRU 门控逻辑
        u = torch.sigmoid(agg_u)  # 更新门
        r = torch.sigmoid(agg_r)  # 重置门

        # 候选状态
        c_input = torch.cat([inputs, r * h_prev], dim=-1)
        val_c_new = self.W_c(c_input)
        agg_c_new = torch.einsum('bnm,bmf->bnf', adj, val_c_new)
        c = torch.tanh(agg_c_new)

        h_new = u * h_prev + (1 - u) * c
        return h_new


class FeatureDecoupler(nn.Module):
    """
    步骤 S2: 客户端因果特征解耦
    包含两个编码器：Encoder_inv (因果) 和 Encoder_spu (干扰/伪相关)
    """

    def __init__(self, input_dim, hidden_dim):
        super(FeatureDecoupler, self).__init__()
        # 假设输入是 (Batch, Nodes, Time, Feat)，这里简化为线性映射提取特征
        # 实际应用中可能使用CNN或LSTM作为编码器
        self.encoder_inv = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder_spu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x shape: [Batch, Nodes, Input_Dim] (假设已经flatten了时间维度或者只取当前步)
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
    利用因果特征生成反对称的动态邻接矩阵
    """

    def __init__(self, hidden_dim, d_model):
        super(CausalGraphGenerator, self).__init__()
        self.W1 = nn.Linear(hidden_dim, d_model)
        self.W2 = nn.Linear(hidden_dim, d_model)

    def forward(self, h_inv):
        # h_inv: [Batch, Nodes, Hidden]

        # 映射到源节点和目标节点空间
        M1 = torch.tanh(self.W1(h_inv))  # [B, N, d]
        M2 = torch.tanh(self.W2(h_inv))  # [B, N, d]

        # 计算反对称差异: M1 * M2^T - M2 * M1^T
        # Einstein summation: bni, bmj -> bnm
        term1 = torch.einsum('bnd,bmd->bnm', M1, M2)
        term2 = torch.einsum('bnd,bmd->bnm', M2, M1)

        diff = term1 - term2

        # 激活函数生成有向图
        A_causal = F.relu(torch.tanh(diff))

        return A_causal


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