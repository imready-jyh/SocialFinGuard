import numpy as np


class SimpleRAG:
    """简化的检索增强生成模块"""

    def __init__(self):
        self.knowledge_base = {
            'rules': [
                "深夜（22:00-06:00）大额交易（>5000）风险高",
                "短时间内高频交易（>10次/小时）可能是洗钱",
                "跨地域异常交易模式需要关注",
                "新账户立即进行大额交易是典型诈骗特征"
            ],
            'fraud_patterns': [
                "金字塔式传销：多层级发展下线",
                "庞氏骗局：用新投资者资金支付旧投资者",
                "刷单欺诈：虚假交易刷高信用"
            ]
        }

    def retrieve_relevant_rules(self, transaction_features):
        """根据交易特征检索相关规则"""
        matched_rules = []

        if transaction_features.get('hour', 12) in [22, 23, 0, 1, 2, 3, 4, 5, 6]:
            if transaction_features.get('amount', 0) > 5000:
                matched_rules.append(self.knowledge_base['rules'][0])

        if transaction_features.get('frequency', 0) > 10:
            matched_rules.append(self.knowledge_base['rules'][1])

        return matched_rules

    def explain_prediction(self, features, prediction):
        """生成可解释的预测结果"""
        rules = self.retrieve_relevant_rules(features)

        explanation = {
            'prediction': 'fraud' if prediction == 1 else 'normal',
            'confidence': np.random.random(),
            'matched_rules': rules,
            'risk_factors': []
        }

        if features.get('hour', 12) < 6 or features.get('hour', 12) > 22:
            explanation['risk_factors'].append('异常交易时间')
        if features.get('amount', 0) > 3000:
            explanation['risk_factors'].append('大额交易')
        if features.get('frequency', 0) > 5:
            explanation['risk_factors'].append('高频交易')

        return explanation