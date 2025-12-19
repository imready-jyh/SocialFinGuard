import numpy as np
import pandas as pd
import networkx as nx
from utils import SEED

np.random.seed(SEED)


class FraudDataGenerator:
    """生成模拟的金融交易和社交网络数据（优化版：特征与诈骗标签关联）"""

    def __init__(self, n_users=1000, n_transactions=5000, fraud_ratio=0.05):
        self.n_users = n_users
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio

    def generate_social_network(self):
        """生成社交网络图（特征与诈骗标签强关联）"""
        # 使用幂律分布生成更真实的社交网络
        G = nx.powerlaw_cluster_graph(self.n_users, m=3, p=0.05)

        # 先计算拓扑特征（为后续特征关联做准备）
        degree_cent = nx.degree_centrality(G)
        between_cent = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        pagerank = nx.pagerank(G)

        # 添加节点特征（拓扑特征+行为特征，与诈骗标签关联）
        for node in G.nodes():
            # 基础特征（非诈骗用户：低中心性、低活跃度）
            base_feat = [
                degree_cent[node],  # 度中心性
                between_cent[node],  # 介数中心性
                clustering[node],  # 聚类系数
                pagerank[node],  # PageRank
                np.random.uniform(0, 1),  # 账户活跃度（正常用户低）
                np.random.uniform(8, 22),  # 活跃时段（正常用户日间）
                np.random.exponential(500),  # 平均交易金额（正常用户低）
                np.random.poisson(2),  # 交易频率（正常用户低）
                np.random.uniform(0, 0.2),  # 跨域交易占比（正常用户低）
                np.random.uniform(0, 0.1)  # 陌生交易占比（正常用户低）
            ]
            G.nodes[node]['features'] = np.array(base_feat)
            G.nodes[node]['is_fraud'] = 0

        # 标记部分节点为诈骗者，并修改其特征（强区分度）
        fraud_users = np.random.choice(
            self.n_users,
            size=int(self.n_users * self.fraud_ratio),
            replace=False
        )
        for user in fraud_users:
            G.nodes[user]['is_fraud'] = 1
            # 诈骗者特征：高中心性、深夜活跃、大额高频、跨域/陌生交易多
            feat = G.nodes[user]['features']
            feat[0] *= 3  # 度中心性×3
            feat[1] *= 5  # 介数中心性×5
            feat[4] = np.random.uniform(0.7, 1.0)  # 账户活跃度高
            feat[5] = np.random.choice([22, 23, 0, 1, 2, 3])  # 深夜活跃
            feat[6] = np.random.exponential(5000)  # 大额交易
            feat[7] = np.random.poisson(10)  # 高频交易
            feat[8] = np.random.uniform(0.7, 1.0)  # 跨域交易占比高
            feat[9] = np.random.uniform(0.8, 1.0)  # 陌生交易占比高
            G.nodes[user]['features'] = feat

            # 诈骗者的邻居也更可能是诈骗者（同步修改特征）
            neighbors = list(G.neighbors(user))
            for neighbor in neighbors[:2]:
                if np.random.random() > 0.7:
                    G.nodes[neighbor]['is_fraud'] = 1
                    n_feat = G.nodes[neighbor]['features']
                    n_feat[0] *= 2
                    n_feat[1] *= 3
                    n_feat[4] = np.random.uniform(0.5, 1.0)
                    G.nodes[neighbor]['features'] = n_feat

        return G

    def generate_transactions(self, G):
        """生成交易数据（与节点特征关联）"""
        transactions = []
        edges = list(G.edges())

        for i in range(self.n_transactions):
            if np.random.random() > 0.3 and len(edges) > 0:
                # 基于社交关系的交易
                src, dst = edges[np.random.randint(len(edges))]
            else:
                # 随机交易
                src, dst = np.random.randint(0, self.n_users, 2)

            is_fraud = G.nodes[src]['is_fraud'] or G.nodes[dst]['is_fraud']

            # 交易特征与节点特征关联（而非纯随机）
            src_feat = G.nodes[src]['features']
            if is_fraud:
                amount = src_feat[6]  # 用诈骗者的大额特征
                hour = src_feat[5]  # 用诈骗者的深夜时段
                frequency = src_feat[7]  # 用诈骗者的高频特征
            else:
                amount = src_feat[6]  # 用正常用户的小额特征
                hour = src_feat[5]  # 用正常用户的日间时段
                frequency = src_feat[7]  # 用正常用户的低频特征

            transactions.append({
                'src': src,
                'dst': dst,
                'amount': amount,
                'hour': hour,
                'frequency': frequency,
                'timestamp': i,
                'is_fraud': int(is_fraud and np.random.random() > 0.2)
            })

        return pd.DataFrame(transactions)