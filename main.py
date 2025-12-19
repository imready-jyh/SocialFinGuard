import numpy as np
import pandas as pd
import networkx as nx
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 导入自定义模块
from gnn_models import MultiViewGNN, BIAN
from trainer import FraudDetectionTrainer
from utils import DEVICE, SEED, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, FRAUD_WEIGHT
from ieee_cis_loader import IEEECISDataLoader  # 新增导入


# 社会传播分析器（保持不变）
class SocialPropagationAnalyzer:
    def __init__(self, G, transactions):
        self.G = G
        self.transactions = transactions
        self.fraud_nodes = [n for n, d in G.nodes(data=True) if d.get('is_fraud', 0) == 1]
        self.normal_nodes = [n for n, d in G.nodes(data=True) if d.get('is_fraud', 0) == 0]

    def analyze_network_topology(self):
        metrics = {}
        # 度中心性
        degree_cent = nx.degree_centrality(self.G)
        metrics['degree_centrality'] = {
            'fraud_mean': np.mean([degree_cent[n] for n in self.fraud_nodes]),
            'normal_mean': np.mean([degree_cent[n] for n in self.normal_nodes]),
            'difference': np.mean([degree_cent[n] for n in self.fraud_nodes]) - np.mean(
                [degree_cent[n] for n in self.normal_nodes])
        }
        # 介数中心性
        between_cent = nx.betweenness_centrality(self.G)
        metrics['betweenness_centrality'] = {
            'fraud_mean': np.mean([between_cent[n] for n in self.fraud_nodes]),
            'normal_mean': np.mean([between_cent[n] for n in self.normal_nodes]),
            'difference': np.mean([between_cent[n] for n in self.fraud_nodes]) - np.mean(
                [between_cent[n] for n in self.normal_nodes])
        }
        # 聚类系数
        clustering = nx.clustering(self.G)
        metrics['clustering_coefficient'] = {
            'fraud_mean': np.mean([clustering[n] for n in self.fraud_nodes]),
            'normal_mean': np.mean([clustering[n] for n in self.normal_nodes]),
            'difference': np.mean([clustering[n] for n in self.fraud_nodes]) - np.mean(
                [clustering[n] for n in self.normal_nodes])
        }
        # PageRank
        pagerank = nx.pagerank(self.G)
        metrics['pagerank'] = {
            'fraud_mean': np.mean([pagerank[n] for n in self.fraud_nodes]),
            'normal_mean': np.mean([pagerank[n] for n in self.normal_nodes]),
            'difference': np.mean([pagerank[n] for n in self.fraud_nodes]) - np.mean(
                [pagerank[n] for n in self.normal_nodes])
        }
        return metrics

    def detect_communities(self):
        # 使用Louvain算法检测社区
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(self.G)
            communities = {}
            for node, comm in partition.items():
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)

            # 计算每个社区的诈骗率
            comm_data = []
            for comm_id, nodes in communities.items():
                fraud_count = sum(1 for n in nodes if n in self.fraud_nodes)
                total_count = len(nodes)
                fraud_ratio = fraud_count / total_count if total_count > 0 else 0
                comm_data.append({
                    'community_id': comm_id,
                    'size': total_count,
                    'fraud_count': fraud_count,
                    'fraud_ratio': fraud_ratio
                })
            return pd.DataFrame(comm_data)
        except ImportError:
            print("  警告：未安装python-louvain，跳过社区检测")
            return pd.DataFrame()


# 主函数（替换数据生成逻辑）
def main():
    print("=" * 60)
    print("基于社会计算的金融诈骗检测系统 (IEEE-CIS Dataset)")
    print("=" * 60)

    # 1. 加载IEEE-CIS数据集（替换原数据生成）
    print("\n[1/8] 加载IEEE-CIS数据集...")

    # ⚠️ 请修改为你的数据路径
    data_loader = IEEECISDataLoader(
        train_transaction_path='./data/train_transaction.csv',  # 你的数据集路径
        train_identity_path='./data/train_identity.csv',  # 可选，无则设为None
        sample_size=20000,  # 建议先小批量测试（如1万条），再逐步增大
        random_state=SEED
    )

    try:
        df = data_loader.load_data()
        df = data_loader.engineer_features()
        G = data_loader.build_graph()
        graph_data = data_loader.build_multi_view_graph()
        transactions = data_loader.get_transactions_dataframe()
    except FileNotFoundError:
        print("\n❌ 错误：找不到数据文件！")
        print("请下载IEEE-CIS数据集并修改路径：")
        print("1. 访问: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("2. 下载 train_transaction.csv 和 train_identity.csv")
        print("3. 放到 ./data/ 目录下或修改代码中的路径")
        return

    # 2. 分析社会传播特征
    print("\n[2/8] 分析社会传播特征...")
    analyzer = SocialPropagationAnalyzer(G, transactions)

    topology = analyzer.analyze_network_topology()
    print("\n  网络拓扑特征对比：")
    for metric, values in topology.items():
        print(f"    {metric}:")
        print(f"      诈骗者均值: {values['fraud_mean']:.4f}")
        print(f"      正常用户均值: {values['normal_mean']:.4f}")
        print(f"      差异: {values['difference']:.4f}")

    communities = analyzer.detect_communities()
    if not communities.empty:
        print(f"\n  检测到 {len(communities)} 个社区")
        high_risk_communities = communities[communities['fraud_ratio'] > 0.05]
        print(f"  高风险社区（诈骗率>5%）: {len(high_risk_communities)} 个")
    else:
        print("\n  未检测到社区（请安装python-louvain: pip install python-louvain）")

    # 3. 准备训练数据
    print("\n[3/8] 准备训练数据...")

    x = graph_data['x'].to(DEVICE)
    y = graph_data['y'].to(DEVICE)
    edge_indices = [ei.to(DEVICE) for ei in graph_data['edge_indices']]
    edge_index = graph_data['edge_index'].to(DEVICE)
    edge_attr = graph_data['edge_attr'].to(DEVICE)
    timestamps = graph_data['timestamps'].to(DEVICE)

    # 划分训练/验证/测试集
    n_nodes = x.size(0)
    indices = np.arange(n_nodes)
    labels_np = y.cpu().numpy()

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels_np, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels_np[temp_idx], random_state=SEED
    )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(DEVICE)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(DEVICE)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(DEVICE)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"  训练集: {train_mask.sum().item()} | 验证集: {val_mask.sum().item()} | 测试集: {test_mask.sum().item()}")
    print(f"  训练集诈骗率: {y[train_mask].float().mean().item():.2%}")

    # 4. 训练MultiViewGNN
    print("\n[4/8] 训练多视图图神经网络...")

    # 特征归一化
    scaler = StandardScaler()
    x_np = x.cpu().numpy()
    x_scaled = torch.FloatTensor(scaler.fit_transform(x_np)).to(DEVICE)

    multi_view_data = {
        'x': x_scaled,
        'edge_indices': edge_indices
    }

    # 适配特征维度（IEEE-CIS特征维度可能不是10）
    node_feature_dim = x_scaled.size(1)
    model_gnn = MultiViewGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        output_dim=2,
        n_views=3
    ).to(DEVICE)
    trainer_gnn = FraudDetectionTrainer(model_gnn)

    train_history_gnn = {'accuracy': [], 'auc': [], 'precision': [], 'recall': []}
    val_history_gnn = {'accuracy': [], 'auc': [], 'precision': [], 'recall': []}

    # 早停逻辑
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        loss = trainer_gnn.train_epoch(multi_view_data, y, train_mask)

        if (epoch + 1) % 10 == 0:  # 每10轮评估一次
            train_metrics = trainer_gnn.evaluate(multi_view_data, y, train_mask)
            val_metrics = trainer_gnn.evaluate(multi_view_data, y, val_mask)

            for key in train_history_gnn:
                train_history_gnn[key].append(train_metrics[key])
                val_history_gnn[key].append(val_metrics[key])

            print(f"  Epoch {epoch + 1}/{EPOCHS} - Loss: {loss:.4f}")
            print(f"    Val AUC: {val_metrics['auc']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")

            # 早停逻辑
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                patience_counter = 0
                # 保存最佳模型
                torch.save(model_gnn.state_dict(), 'best_gnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（Epoch {epoch + 1}），最佳Val AUC: {best_auc:.4f}")
                    break

    # 5. 训练行为信息聚合网络...
    print("\n[5/8] 训练行为信息聚合网络...")

    # 适配边特征维度
    edge_dim = edge_attr.size(1)
    bian_data = {
        'x': x_scaled,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        # 修复：时间戳已在loader中转为LongTensor，直接传入
        'timestamps': timestamps
    }

    model_bian = BIAN(
        node_dim=node_feature_dim,
        edge_dim=edge_dim,
        hidden_dim=64
    ).to(DEVICE)
    trainer_bian = FraudDetectionTrainer(model_bian)

    # 优化：添加早停+动态权重
    best_bian_auc = 0.0
    patience_counter = 0
    for epoch in range(EPOCHS):
        loss = trainer_bian.train_epoch(bian_data, y, train_mask)

        if (epoch + 1) % 10 == 0:
            val_metrics = trainer_bian.evaluate(bian_data, y, val_mask)
            print(f"  Epoch {epoch + 1}/{EPOCHS} - Loss: {loss:.4f}")
            print(f"    Val AUC: {val_metrics['auc']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f}")

            if val_metrics['auc'] > best_bian_auc:
                best_bian_auc = val_metrics['auc']
                patience_counter = 0
                torch.save(model_bian.state_dict(), 'best_bian_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:  # 早停阈值
                    print(f"  早停触发（Epoch {epoch + 1}），最佳Val AUC: {best_bian_auc:.4f}")
                    break

    # 6. 测试集评估
    print("\n[6/8] 测试集最终评估...")

    # 评估MultiViewGNN
    test_metrics_gnn = trainer_gnn.evaluate(multi_view_data, y, test_mask)
    print("\n  MultiViewGNN 性能:")
    for key, value in test_metrics_gnn.items():
        print(f"    {key}: {value:.4f}")

    # 评估BIAN
    test_metrics_bian = trainer_bian.evaluate(bian_data, y, test_mask)
    print("\n  BIAN 性能:")
    for key, value in test_metrics_bian.items():
        print(f"    {key}: {value:.4f}")

    # 7. 生成可解释预测
    print("\n[7/8] 生成可解释预测...")

    # 加载最佳模型
    model_gnn.load_state_dict(torch.load('best_gnn_model.pth'))
    model_gnn.eval()

    # 随机选5个样本
    sample_indices = np.random.choice(test_idx, size=min(5, len(test_idx)), replace=False)
    print("\n  样本预测解释:")
    for idx in sample_indices:
        with torch.no_grad():
            logits, _ = model_gnn(x_scaled, edge_indices)
            pred = logits.argmax(dim=1)[idx].item()
            true = y[idx].item()
            risk_score = torch.softmax(logits, dim=1)[idx][1].item()

            # 风险因素分析
            risk_factors = []
            node_feat = x_np[idx]
            # 交易金额异常
            if node_feat[data_loader.feature_cols.index('TransactionAmt')] > 2:
                risk_factors.append('大额交易')
            # 交易频率异常
            if transactions.iloc[idx]['frequency'] > 5:
                risk_factors.append('高频交易')
            # 深夜交易
            if transactions.iloc[idx]['hour'] in [22, 23, 0, 1, 2, 3]:
                risk_factors.append('深夜交易')

            print(f"\n  用户 {idx}:")
            print(f"    预测: {'fraud' if pred == 1 else 'normal'} (真实标签: {'fraud' if true == 1 else 'normal'})")
            print(f"    风险分数: {risk_score:.3f}")
            print(f"    匹配规则: {[]}")  # 可扩展业务规则
            print(f"    风险因素: {risk_factors if risk_factors else []}")

    # 8. 设计干预策略
    print("\n[8/8] 设计干预策略...")

    # 高风险用户（风险分数>0.8）
    with torch.no_grad():
        logits, _ = model_gnn(x_scaled, edge_indices)
        risk_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        high_risk_users = [i for i in range(len(risk_scores)) if risk_scores[i] > 0.8 and test_mask[i]]

    print(f"\n  发出 {len(high_risk_users)} 个高风险预警")
    if high_risk_users:
        top_risk = np.argmax(risk_scores[high_risk_users])
        top_idx = high_risk_users[top_risk]
        print(f"    示例: 用户 {top_idx} - 风险分数 {risk_scores[top_idx]:.3f} - 建议: 账户冻结")

    # 关键传播节点（介数中心性高的诈骗节点）
    between_cent = nx.betweenness_centrality(G)
    fraud_nodes = [n for n in G.nodes() if G.nodes[n]['is_fraud'] == 1]
    key_nodes = sorted(fraud_nodes, key=lambda x: between_cent[x], reverse=True)[:15]
    print(f"\n  识别 {len(key_nodes)} 个关键传播节点需要阻断")

    # 教育提示用户（中风险）
    medium_risk_users = [i for i in range(len(risk_scores)) if 0.3 < risk_scores[i] <= 0.8 and test_mask[i]]
    print(f"\n  向 {len(medium_risk_users)} 个用户发送教育提示")

    # 9. 生成可视化结果
    print("\n[9/9] 生成可视化结果...")

    # 训练曲线
    plt.figure(figsize=(12, 4))

    # GNN训练曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_history_gnn['auc'], label='Train AUC')
    plt.plot(val_history_gnn['auc'], label='Val AUC')
    plt.title('MultiViewGNN Training Curve')
    plt.xlabel('Epoch (×5)')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    # 网络拓扑可视化（简化）
    plt.subplot(1, 2, 2)
    # 随机选200个节点可视化
    vis_nodes = np.random.choice(list(G.nodes()), size=min(200, len(G.nodes())), replace=False)
    vis_G = G.subgraph(vis_nodes)
    pos = nx.spring_layout(vis_G)
    # 颜色：诈骗=红，正常=蓝
    node_colors = ['red' if G.nodes[n]['is_fraud'] == 1 else 'blue' for n in vis_G.nodes()]
    nx.draw(vis_G, pos, node_color=node_colors, node_size=50, alpha=0.6)
    plt.title('Network Topology (Fraud=Red, Normal=Blue)')

    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.close()

    # 网络图单独保存
    plt.figure(figsize=(8, 8))
    nx.draw(vis_G, pos, node_color=node_colors, node_size=50, alpha=0.6)
    plt.savefig('network.png')
    plt.close()

    print("  ✓ 网络图已保存: network.png")
    print("  ✓ 训练曲线已保存: metrics.png")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)

    return {
        'gnn_metrics': test_metrics_gnn,
        'bian_metrics': test_metrics_bian,
        'high_risk_users': high_risk_users,
        'key_nodes': key_nodes
    }


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # 运行主程序
    results = main()