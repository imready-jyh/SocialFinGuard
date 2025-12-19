import numpy as np
import pandas as pd
import networkx as nx


class SocialPropagationAnalyzer:
    """分析诈骗在社交网络中的传播特征"""

    def __init__(self, G, transactions):
        self.G = G
        self.transactions = transactions

    def analyze_network_topology(self):
        """网络拓扑特征分析"""
        metrics = {
            'degree_centrality': nx.degree_centrality(self.G),
            'betweenness_centrality': nx.betweenness_centrality(self.G),
            'clustering_coefficient': nx.clustering(self.G),
            'pagerank': nx.pagerank(self.G)
        }

        # 对比诈骗者与正常用户的网络特征
        fraud_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['is_fraud']]
        normal_nodes = [n for n in self.G.nodes() if not self.G.nodes[n]['is_fraud']]

        comparison = {}
        for metric_name, metric_values in metrics.items():
            fraud_values = [metric_values[n] for n in fraud_nodes]
            normal_values = [metric_values[n] for n in normal_nodes]
            comparison[metric_name] = {
                'fraud_mean': np.mean(fraud_values),
                'normal_mean': np.mean(normal_values),
                'difference': np.mean(fraud_values) - np.mean(normal_values)
            }

        return comparison

    def detect_communities(self):
        """社区检测与诈骗聚集分析"""
        communities = nx.community.greedy_modularity_communities(self.G)

        community_stats = []
        for i, community in enumerate(communities):
            fraud_count = sum(1 for node in community if self.G.nodes[node]['is_fraud'])
            community_stats.append({
                'community_id': i,
                'size': len(community),
                'fraud_count': fraud_count,
                'fraud_ratio': fraud_count / len(community)
            })

        return pd.DataFrame(community_stats)

    def temporal_propagation_analysis(self):
        """时序传播模式分析"""
        fraud_txs = self.transactions[self.transactions['is_fraud'] == 1].sort_values('timestamp')

        # 计算诈骗交易的时间间隔分布
        if len(fraud_txs) > 1:
            time_diffs = fraud_txs['timestamp'].diff().dropna()
            return {
                'mean_interval': time_diffs.mean(),
                'std_interval': time_diffs.std(),
                'burst_detection': (time_diffs < time_diffs.mean() / 2).sum()
            }
        return {}