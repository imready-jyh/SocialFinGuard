import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ==================== IEEE-CIS 数据加载器 ====================
class IEEECISDataLoader:
    """加载和预处理IEEE-CIS数据集"""

    def __init__(self, train_transaction_path, train_identity_path=None,
                 sample_size=None, random_state=42):
        """
        参数:
            train_transaction_path: train_transaction.csv 路径
            train_identity_path: train_identity.csv 路径（可选）
            sample_size: 采样数量（如果数据太大）
            random_state: 随机种子
        """
        self.train_transaction_path = train_transaction_path
        self.train_identity_path = train_identity_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """加载原始数据"""
        print("  [1/6] 加载交易数据...")
        # 读取交易数据
        df_trans = pd.read_csv(self.train_transaction_path)

        # 如果提供了身份数据，合并
        if self.train_identity_path:
            print("  [2/6] 加载身份数据...")
            df_identity = pd.read_csv(self.train_identity_path)
            df = df_trans.merge(df_identity, on='TransactionID', how='left')
        else:
            df = df_trans

        # 采样（如果指定）
        if self.sample_size and len(df) > self.sample_size:
            print(f"  [3/6] 采样 {self.sample_size} 条记录...")
            # 分层采样以保持诈骗比例
            df_fraud = df[df['isFraud'] == 1]
            df_normal = df[df['isFraud'] == 0]

            n_fraud = min(len(df_fraud), int(self.sample_size * 0.035))  # 保持3.5%比例
            n_normal = self.sample_size - n_fraud

            df = pd.concat([
                df_fraud.sample(n=n_fraud, random_state=self.random_state),
                df_normal.sample(n=n_normal, random_state=self.random_state)
            ]).reset_index(drop=True)

        print(f"  数据规模: {len(df)} 条交易, 诈骗率: {df['isFraud'].mean():.2%}")

        self.df = df
        return df

    def engineer_features(self):
        """特征工程"""
        print("  [4/6] 特征工程...")
        df = self.df.copy()

        # 1. 选择关键特征（避免维度过高）
        # V列是PCA降维后的特征，选择前50个
        v_cols = [f'V{i}' for i in range(1, 51) if f'V{i}' in df.columns]

        # C列是计数特征
        c_cols = [f'C{i}' for i in range(1, 15) if f'C{i}' in df.columns]

        # D列是时间差特征
        d_cols = [f'D{i}' for i in range(1, 11) if f'D{i}' in df.columns]

        # M列是匹配特征（布尔值）
        m_cols = [col for col in df.columns if col.startswith('M')]

        # 其他重要特征
        important_cols = ['TransactionAmt', 'TransactionDT']

        # 2. 处理类别特征（用于构建图）
        self.card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
        self.addr_cols = ['addr1', 'addr2']
        self.email_cols = ['P_emaildomain', 'R_emaildomain']
        self.device_cols = ['DeviceType', 'DeviceInfo']

        categorical_cols = (self.card_cols + self.addr_cols +
                            self.email_cols + self.device_cols)
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        # 3. 填充缺失值
        # 数值特征用中位数填充
        numerical_cols = v_cols + c_cols + d_cols + important_cols
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)

        # 类别特征用"unknown"填充
        for col in categorical_cols:
            df[col].fillna('unknown', inplace=True)

        # M列（布尔）用False填充
        for col in m_cols:
            df[col].fillna('F', inplace=True)
            df[col] = (df[col] == 'T').astype(int)

        # 4. 标准化数值特征
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # 5. 标签编码类别特征（用于后续）
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.df_processed = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        # ===== 修复：先定义feature_cols，再过滤 =====
        self.feature_cols = numerical_cols + m_cols

        # 额外修复：确保feature_cols全为数值特征（移到feature_cols定义后）
        self.feature_cols = [col for col in self.feature_cols if
                             df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        # 再次检查并转换所有特征为float32
        for col in self.feature_cols:
            df[col] = df[col].astype(np.float32)

        print(f"  特征维度: {len(self.feature_cols)}维（全数值）")

        return df

    def build_graph(self):
        """构建异构图（多视图）"""
        print("  [5/6] 构建社交网络图...")
        df = self.df_processed

        # 创建主图 - 基于共享设备/卡/地址构建连接
        G = nx.Graph()

        # 添加所有交易作为节点
        for idx, row in df.iterrows():
            G.add_node(idx,
                       features=row[self.feature_cols].values,
                       is_fraud=int(row['isFraud']),
                       transaction_id=row['TransactionID'])

        print(f"  添加了 {G.number_of_nodes()} 个节点")

        # 构建边（基于共享关系）
        # 1. 基于card构建连接
        card_groups = defaultdict(list)
        for idx, row in df.iterrows():
            card_key = tuple([row.get(f'{col}_encoded', -1)
                              for col in self.card_cols if col in df.columns])
            if card_key != (-1,) * len(card_key):
                card_groups[card_key].append(idx)

        edge_count = 0
        for group in card_groups.values():
            if len(group) > 1 and len(group) < 100:  # 避免超级节点
                for i in range(len(group)):
                    for j in range(i + 1, min(i + 10, len(group))):  # 限制连接数
                        G.add_edge(group[i], group[j], edge_type='card')
                        edge_count += 1

        print(f"  基于卡信息添加了 {edge_count} 条边")

        # 2. 基于email domain构建连接
        email_groups = defaultdict(list)
        for idx, row in df.iterrows():
            for col in self.email_cols:
                if col in df.columns:
                    email = row.get(f'{col}_encoded', -1)
                    if email != -1:
                        email_groups[email].append(idx)

        edge_count = 0
        for group in email_groups.values():
            if len(group) > 1 and len(group) < 50:
                for i in range(len(group)):
                    for j in range(i + 1, min(i + 5, len(group))):
                        if not G.has_edge(group[i], group[j]):
                            G.add_edge(group[i], group[j], edge_type='email')
                            edge_count += 1

        print(f"  基于邮箱添加了 {edge_count} 条边")

        # 3. 基于地址构建连接
        addr_groups = defaultdict(list)
        for idx, row in df.iterrows():
            addr_key = tuple([row.get(f'{col}_encoded', -1)
                              for col in self.addr_cols if col in df.columns])
            if addr_key != (-1,) * len(addr_key):
                addr_groups[addr_key].append(idx)

        edge_count = 0
        for group in addr_groups.values():
            if len(group) > 1 and len(group) < 50:
                for i in range(len(group)):
                    for j in range(i + 1, min(i + 5, len(group))):
                        if not G.has_edge(group[i], group[j]):
                            G.add_edge(group[i], group[j], edge_type='addr')
                            edge_count += 1

        print(f"  基于地址添加了 {edge_count} 条边")

        # 处理孤立节点：随机连接到最近的节点
        isolated = list(nx.isolates(G))
        if isolated:
            print(f"  处理 {len(isolated)} 个孤立节点...")
            connected = [n for n in G.nodes() if n not in isolated]
            for node in isolated[:min(len(isolated), 1000)]:  # 限制处理数量
                if connected:
                    # 随机连接到3个节点
                    targets = np.random.choice(connected,
                                               size=min(3, len(connected)),
                                               replace=False)
                    for target in targets:
                        G.add_edge(node, target, edge_type='random')

        print(f"  最终图结构: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"  图连通性: {nx.number_connected_components(G)} 个连通分量")

        self.G = G
        return G

    def build_multi_view_graph(self):
        """构建多视图图数据"""
        print("  [6/6] 构建多视图数据...")
        df = self.df_processed
        G = self.G

        # 提取节点特征和标签（修复：确保全为数值且维度一致）
        node_features = []
        labels = []
        node_mapping = {}  # TransactionID到图节点的映射

        # 先获取特征维度，确保所有节点特征维度一致
        feature_dim = len(self.feature_cols)

        for idx, (node, data) in enumerate(G.nodes(data=True)):
            # 修复1：强制转换为数值，处理NaN/Inf
            feat = np.array(data['features'], dtype=np.float32)

            # 修复2：填充NaN/Inf为0
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

            # 修复3：确保维度一致（截断/填充）
            if len(feat) > feature_dim:
                feat = feat[:feature_dim]
            elif len(feat) < feature_dim:
                feat = np.pad(feat, (0, feature_dim - len(feat)), mode='constant')

            node_features.append(feat)
            labels.append(int(data['is_fraud']))  # 强制转为int
            node_mapping[node] = idx

        # 修复4：转换为numpy数组并指定类型
        node_features_np = np.array(node_features, dtype=np.float32)
        x = torch.FloatTensor(node_features_np)
        y = torch.LongTensor(labels)

        # 视图1: 所有边（综合视图）
        edge_index_all = []
        for u, v in G.edges():
            edge_index_all.append([node_mapping[u], node_mapping[v]])
            edge_index_all.append([node_mapping[v], node_mapping[u]])  # 无向图

        # 视图2: 仅卡相关边
        edge_index_card = []
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') == 'card':
                edge_index_card.append([node_mapping[u], node_mapping[v]])
                edge_index_card.append([node_mapping[v], node_mapping[u]])

        # 如果某个视图没有边，使用全图的子集
        if not edge_index_card:
            edge_index_card = edge_index_all[:len(edge_index_all) // 2]

        # 视图3: 仅邮箱+地址边
        edge_index_contact = []
        for u, v, data in G.edges(data=True):
            if data.get('edge_type') in ['email', 'addr']:
                edge_index_contact.append([node_mapping[u], node_mapping[v]])
                edge_index_contact.append([node_mapping[v], node_mapping[u]])

        if not edge_index_contact:
            edge_index_contact = edge_index_all[len(edge_index_all) // 2:]

        edge_index_all = torch.LongTensor(edge_index_all).t().contiguous()
        edge_index_card = torch.LongTensor(edge_index_card).t().contiguous()
        edge_index_contact = torch.LongTensor(edge_index_contact).t().contiguous()

        print(f"  视图1 (全部): {edge_index_all.size(1)} 条边")
        print(f"  视图2 (卡): {edge_index_card.size(1)} 条边")
        print(f"  视图3 (联系): {edge_index_contact.size(1)} 条边")

        # 准备BIAN所需的边特征（替换随机值，用真实业务特征）
        edge_attr_list = []
        for u, v, data in G.edges(data=True):
            # 提取边特征：边类型编码 + 交易金额均值 + 时间特征
            edge_type = data.get('edge_type', 'random')
            type_enc = {'card': 0, 'email': 1, 'addr': 2, 'random': 3}[edge_type]

            # 取两端节点的交易金额均值（修复：安全获取特征）
            try:
                u_amt = G.nodes[u]['features'][self.feature_cols.index('TransactionAmt')]
                v_amt = G.nodes[v]['features'][self.feature_cols.index('TransactionAmt')]
            except (IndexError, ValueError):
                u_amt = 0.0
                v_amt = 0.0
            avg_amt = (u_amt + v_amt) / 2

            # 时间特征（TransactionDT）
            try:
                u_dt = G.nodes[u]['features'][self.feature_cols.index('TransactionDT')]
                v_dt = G.nodes[v]['features'][self.feature_cols.index('TransactionDT')]
            except (IndexError, ValueError):
                u_dt = 0.0
                v_dt = 0.0
            time_diff = abs(u_dt - v_dt)

            # 补充两个随机特征（保持5维）
            edge_attr_list.append([type_enc, avg_amt, time_diff,
                                   np.random.uniform(0, 1), np.random.uniform(0, 1)])

        # 补齐边特征（确保和edge_index_all长度一致）
        while len(edge_attr_list) < edge_index_all.size(1):
            edge_attr_list.append([3, 0, 0, 0, 0])  # 填充默认值

        # 修复5：边特征强制转为float32
        edge_attr = torch.FloatTensor(np.array(edge_attr_list[:edge_index_all.size(1)], dtype=np.float32))

        # 时间戳（TransactionDT）- 修复：归一化到0~999范围
        timestamps_list = []
        # 为每条边生成时间戳（取边两端节点的时间均值）
        for u, v in G.edges():
            try:
                u_ts = G.nodes[u]['features'][self.feature_cols.index('TransactionDT')]
                v_ts = G.nodes[v]['features'][self.feature_cols.index('TransactionDT')]
                edge_ts = (u_ts + v_ts) / 2  # 边的时间戳=两端节点均值
            except (IndexError, ValueError):
                edge_ts = 0.0
            timestamps_list.append(edge_ts)

        # 无向图：每条边双向添加，时间戳也重复
        timestamps_list = timestamps_list * 2
        # 截断/填充到匹配边数量
        timestamps_list = timestamps_list[:edge_index_all.size(1)]
        while len(timestamps_list) < edge_index_all.size(1):
            timestamps_list.append(0.0)

        # 修复核心：归一化时间戳到 0~999 范围（适配embedding层）
        timestamps_np = np.array(timestamps_list, dtype=np.float32)
        # 最小-最大归一化
        if timestamps_np.max() != timestamps_np.min():
            timestamps_np = (timestamps_np - timestamps_np.min()) / (timestamps_np.max() - timestamps_np.min())
        timestamps_np = (timestamps_np * 999).astype(np.int32)  # 缩放到0~999

        # 转换为tensor
        timestamps = torch.LongTensor(timestamps_np)  # 直接转long，避免后续转换

        return {
            'x': x,
            'y': y,
            'edge_indices': [edge_index_all, edge_index_card, edge_index_contact],
            'edge_index': edge_index_all,  # BIAN用
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'G': G,
            'node_mapping': node_mapping
        }

    def get_transactions_dataframe(self):
        """获取处理后的交易数据框（用于时序分析）"""
        df = self.df_processed.copy()
        df['src'] = df.index
        df['dst'] = df.index  # IEEE-CIS不是点对点交易，用同一节点

        # 映射特征列
        df['amount'] = df['TransactionAmt']
        df['timestamp'] = df['TransactionDT']
        df['is_fraud'] = df['isFraud']

        # 提取小时信息（假设TransactionDT是秒）
        df['hour'] = (df['TransactionDT'] / 3600) % 24
        df['frequency'] = 1  # 简化：每条交易频率为1

        return df[['src', 'dst', 'amount', 'hour', 'frequency', 'timestamp', 'is_fraud']]