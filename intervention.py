import numpy as np
import networkx as nx


class InterventionStrategy:
    """诈骗预防干预策略"""

    def __init__(self, G, model, threshold=0.7):
        self.G = G
        self.model = model
        self.threshold = threshold

    def early_warning(self, risk_scores):
        """早期预警机制"""
        high_risk_users = np.where(risk_scores > self.threshold)[0]

        warnings = []
        for user in high_risk_users:
            warnings.append({
                'user_id': user,
                'risk_score': risk_scores[user],
                'warning_level': 'high' if risk_scores[user] > 0.9 else 'medium',
                'recommended_action': '账户冻结' if risk_scores[user] > 0.9 else '增强监控'
            })

        return warnings

    def propagation_blocking(self, fraud_nodes):
        """传播阻断策略"""
        # 识别关键传播节点
        betweenness = nx.betweenness_centrality(self.G)

        blocking_strategy = []
        for node in fraud_nodes:
            if node in betweenness and betweenness[node] > 0.01:
                neighbors = list(self.G.neighbors(node))
                blocking_strategy.append({
                    'fraud_node': node,
                    'betweenness': betweenness[node],
                    'affected_neighbors': len(neighbors),
                    'action': '断开关键连接'
                })

        return blocking_strategy

    def user_education(self, risk_scores):
        """用户教育与提示"""
        education_targets = np.where((risk_scores > 0.3) & (risk_scores < 0.7))[0]

        messages = []
        for user in education_targets:
            messages.append({
                'user_id': user,
                'message': '检测到您的交易行为存在异常模式，请注意防范诈骗',
                'tips': [
                    '避免深夜进行大额交易',
                    '警惕陌生人要求转账',
                    '定期检查账户异常活动'
                ]
            })

        return messages