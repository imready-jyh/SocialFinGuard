import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import scatter


class NodeLevelAttention(nn.Module):
    """节点级注意力机制"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=4, concat=False)

    def forward(self, x, edge_index):
        return F.elu(self.gat(x, edge_index))


class ViewLevelAttention(nn.Module):
    """视图级注意力融合"""

    def __init__(self, embedding_dim, n_views):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
        self.n_views = n_views

    def forward(self, view_embeddings):
        # view_embeddings: list of [batch_size, embedding_dim]
        stacked = torch.stack(view_embeddings, dim=1)  # [batch, n_views, dim]
        attention_scores = self.attention(stacked).squeeze(-1)  # [batch, n_views]
        attention_weights = F.softmax(attention_scores, dim=1)

        # 加权融合
        fused = torch.sum(stacked * attention_weights.unsqueeze(-1), dim=1)
        return fused, attention_weights


class MultiViewGNN(nn.Module):
    """多视图图神经网络（基于SemiGNN思想）"""

    def __init__(self, node_feature_dim, hidden_dim, output_dim, n_views=3):
        super().__init__()
        self.n_views = n_views

        # 每个视图的节点级注意力层
        self.view_encoders = nn.ModuleList([
            NodeLevelAttention(node_feature_dim, hidden_dim)
            for _ in range(n_views)
        ])

        # 视图级注意力融合
        self.view_fusion = ViewLevelAttention(hidden_dim, n_views)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_indices):
        """
        x: 节点特征 [n_nodes, feature_dim]
        edge_indices: list of edge_index for each view
        """
        # 每个视图生成嵌入
        view_embeddings = []
        for i, encoder in enumerate(self.view_encoders):
            view_emb = encoder(x, edge_indices[i])
            view_embeddings.append(view_emb)

        # 视图融合
        fused_embedding, attention_weights = self.view_fusion(view_embeddings)

        # 分类
        logits = self.classifier(fused_embedding)
        return logits, attention_weights


class EdgeToNodeConv(nn.Module):
    """边到节点卷积层（ETNConv）"""

    def __init__(self, edge_dim, node_dim):
        super().__init__()
        self.edge_transform = nn.Linear(edge_dim, node_dim)

    def forward(self, edge_features, edge_index, n_nodes):
        """
        edge_features: [n_edges, edge_dim]
        edge_index: [2, n_edges]
        """
        # 转换边特征
        transformed_edges = self.edge_transform(edge_features)

        # 聚合到节点
        node_features = torch.zeros(n_nodes, transformed_edges.size(1)).to(edge_features.device)
        src, dst = edge_index

        # 源节点聚合
        node_features.index_add_(0, src, transformed_edges)
        # 目标节点聚合
        node_features.index_add_(0, dst, transformed_edges)

        return node_features


class TemporalEncoder(nn.Module):
    """时序行为编码器"""

    def __init__(self, time_dim):
        super().__init__()
        self.time_embedding = nn.Linear(1, time_dim)

    def forward(self, timestamps):
        """Time2Vec风格的时间编码"""
        t = timestamps.unsqueeze(-1).float()
        return torch.sin(self.time_embedding(t))


class BIAN(nn.Module):
    """行为信息聚合网络"""

    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super(BIAN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # 节点特征编码
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 边特征编码
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 修复：时序嵌入层维度固定为1000（匹配归一化后的索引）
        self.time_emb = nn.Embedding(1000, hidden_dim)  # 0~999索引
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 图卷积层
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 残差连接
        self.residual = nn.Linear(hidden_dim, hidden_dim)

        # 分类头（融合特征维度=4*hidden_dim）
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index, edge_attr, timestamps):
        # 节点特征编码
        x_enc = self.node_encoder(x)
        # 边特征编码
        edge_enc = self.edge_encoder(edge_attr)

        # 时序编码（兼容低版本）
        try:
            time_emb = self.time_emb(timestamps)
        except IndexError:
            timestamps_clamped = torch.clamp(timestamps, 0, 999)
            time_emb = self.time_emb(timestamps_clamped)
        time_enc = self.time_encoder(time_emb)

        # 图卷积+残差连接
        x1 = self.conv1(x_enc, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2 + self.residual(x_enc))

        # 融合特征（维度统一为 [num_edges, hidden_dim]）
        src, dst = edge_index
        src_feat = x2[src]
        dst_feat = x2[dst]

        # 安全检查：确保所有特征维度匹配
        assert src_feat.size(0) == edge_enc.size(0) == time_enc.size(0), \
            f"维度不匹配: src={src_feat.size(0)}, edge={edge_enc.size(0)}, time={time_enc.size(0)}"

        fusion_feat = torch.cat([src_feat, dst_feat, edge_enc, time_enc], dim=-1)

        # ===== 优化：使用scatter_add实现高效节点聚合 =====
        node_fusion = torch.zeros((x.size(0), fusion_feat.size(1)),
                                  device=x.device, dtype=fusion_feat.dtype)
        node_count = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)

        # 对源节点聚合（高效版本）
        node_fusion.scatter_add_(0, src.unsqueeze(1).expand_as(fusion_feat), fusion_feat)
        node_count.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))

        # 避免除以0
        node_count = torch.clamp(node_count, min=1.0)

        # 计算均值
        node_fusion = node_fusion / node_count.unsqueeze(1)

        # 填充空节点（无边的节点）
        mask_empty = (node_fusion.abs().sum(dim=1) < 1e-6)
        if mask_empty.any():
            # 扩展x_enc到匹配维度（4倍）
            repeat_times = fusion_feat.size(1) // x_enc.size(1)
            if fusion_feat.size(1) % x_enc.size(1) != 0:
                repeat_times += 1
            x_enc_expanded = x_enc.repeat(1, repeat_times)[:, :fusion_feat.size(1)]
            node_fusion[mask_empty] = x_enc_expanded[mask_empty]

        # 分类预测
        out = self.classifier(node_fusion)
        return out