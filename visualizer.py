import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


class FraudVisualizer:
    """结果可视化"""

    @staticmethod
    def plot_network(G, fraud_nodes, save_path='network.png'):
        """可视化社交网络"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        colors = ['red' if G.nodes[n]['is_fraud'] else 'lightblue'
                  for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.2)

        plt.title('Financial Social Network (Red: Fraud, Blue: Normal)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_metrics(train_metrics, val_metrics, save_path='metrics.png'):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = ['accuracy', 'auc', 'precision', 'recall']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            ax.plot(train_metrics[metric], label='Train', marker='o')
            ax.plot(val_metrics[metric], label='Val', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()