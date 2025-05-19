
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import math
from collections import defaultdict
def idd(x):
    return x

def build_relation_graph(file_path, n_rel):
    """
    构建关系图
    :param file_path: train.txt 文件的路径
    :param n_rel: 关系的总数
    :return: relation_graph (adjacency list), relation_edges (edge types), relation_weights (weights of edges)
    """
    # Step 1: Load train data from file
    train_data = []
    with open(file_path, 'r') as f:
        for line in f:
            s, r, o = line.strip().split()[:3]
            train_data.append((int(s), int(r), int(o)))

    # Step 2: Collect neighbors for each relation
    relation_neighbors = defaultdict(lambda: {'head': set(), 'tail': set()})
    for s, r, o in train_data:
        relation_neighbors[r]['head'].add(s)  # 添加头实体
        relation_neighbors[r]['tail'].add(o)  # 添加尾实体

    # Step 3: Build the relation graph and compute edge types and weights
    relation_graph = defaultdict(set)
    relation_edges = defaultdict(list)
    relation_weights = defaultdict(float)
    for r1 in range(n_rel):
        for r2 in range(n_rel):
            if r1 != r2:
                # T2T: 尾对尾
                common_heads = relation_neighbors[r1]['head'].intersection(relation_neighbors[r2]['head'])
                if common_heads:
                    relation_graph[r1].add(r2)
                    relation_edges[(r1, r2)].append('T2T')

                # H2H: 头对头
                common_tails = relation_neighbors[r1]['tail'].intersection(relation_neighbors[r2]['tail'])
                if common_tails:
                    relation_graph[r1].add(r2)
                    relation_edges[(r1, r2)].append('H2H')

                # SEQ: 顺序连接
                common_seq = relation_neighbors[r1]['tail'].intersection(relation_neighbors[r2]['head'])
                if common_seq:
                    relation_graph[r1].add(r2)
                    relation_edges[(r1, r2)].append('SEQ')

                # 计算权重
                common_entities = relation_neighbors[r1]['head'].union(relation_neighbors[r1]['tail']).intersection(
                    relation_neighbors[r2]['head'].union(relation_neighbors[r2]['tail']))
                # 防止除以零
                denominator1 = len(relation_neighbors[r1]['head'].union(relation_neighbors[r1]['tail']))
                denominator2 = len(relation_neighbors[r2]['head'].union(relation_neighbors[r2]['tail']))

                if denominator1 > 0 and denominator2 > 0:
                    weight = (len(common_entities) / denominator1) + (len(common_entities) / denominator2)
                    relation_weights[(r1, r2)] = weight

    return relation_graph, relation_edges, relation_weights


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attention_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))  # e.shape: (N * N,)

        # Reshape e to match the dimensions of adj
        e = e.view(adj.size(0), adj.size(1))  # e.shape: (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Now dimensions match
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attention_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class TGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, window_size=5):
        super(TGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.time_embed = nn.Embedding(window_size+1, in_dim//4)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim//4*5, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concat([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]
        # hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    
class GNNLayer2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd):
        super(GNNLayer2, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    


class TimelineGNNLayer(torch.nn.Module):
    """
    设置了自由的时间维度，使用gate、sigmoid，只使用相对时间编码
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(TimelineGNNLayer6, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        ##############################
        # 构建关系图
        # 构建关系图并编码
        file_path = './data/ICEWS18/train.txt'

        relation_graph, relation_edges, relation_weights = build_relation_graph(file_path, n_rel)

        # 将关系图的邻接表转换为邻接矩阵
        num_nodes = n_rel
        adj_matrix = torch.zeros((num_nodes, num_nodes))

        for r1, neighbors in relation_graph.items():
            for r2 in neighbors:
                adj_matrix[r1, r2] = relation_weights[(r1, r2)]

        # 假设我们有关系的初始特征表示
        relation_features = torch.randn(num_nodes, in_dim)  # 初始特征维度



        # 创建 GAT 层
        gat_layer1 = GATLayer(in_features=in_dim, out_features=in_dim)  # 第一层 GAT
        gat_layer2 = GATLayer(in_features=in_dim, out_features=in_dim)  # 第二层 GAT

        # 前向传播，更新关系的特征表示
        relation_features = gat_layer1(relation_features, adj_matrix)
        relation_features = gat_layer2(relation_features, adj_matrix)
        # 使用 GAT 编码得到的关系嵌入
        # 定义线性变换层，用于生成反向关系的嵌入
        reverse_relation_transform = nn.Linear(in_dim, in_dim)

        # 应用线性变换生成反向关系的嵌入
        reverse_relation_features = reverse_relation_transform(relation_features)

        # 添加一个特殊的“无关系”或“默认关系”嵌入
        special_relation_feature = torch.zeros(1, in_dim)  # (1, in_dim)

        # 合并所有特征
        expanded_relation_features = torch.cat([
            relation_features,  # 正向关系
            reverse_relation_features,  # 反向关系
            special_relation_feature  # 特殊关系
        ], dim=0)

        # 使用预训练的嵌入
        self.rela_embed = nn.Embedding.from_pretrained(expanded_relation_features, freeze=False)
        #self.rela_embed = nn.Embedding.from_pretrained(relation_features, freeze=False)
        ##############################

        #self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.time_embed = TimeEncoding(hidden_dim=self.time_dim,max_length=max_history_length)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        # Encode relation embeddings using GAT



        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        temp_time_emb = self.time_embed(output[:,1])
        temp_comp_raw = torch.concat([temp_rel_emb,temp_time_emb],dim=1)
        temp_comp = self.fuse_mlp(temp_comp_raw)+temp_rel_emb
        hr = temp_comp[reverse_indexes]
        hs = hidden[sub]


        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = self.gate(hr, h_qr, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new
    

    
class GNNLayer6(torch.nn.Module):
    """
    设置了自由的时间维度，使用gate、sigmoid，只使用相对时间编码
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=idd, max_history_length=1000,time_dim=-1):
        super(GNNLayer6, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        if time_dim<=0:
            self.time_dim = in_dim//4
        else:
            self.time_dim = time_dim
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.fuse_mlp = nn.Sequential(nn.Linear(in_dim+self.time_dim, in_dim),nn.LeakyReLU(),nn.Linear(in_dim, in_dim),nn.LeakyReLU())
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1,bias=False)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node):
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        output, reverse_indexes = edges[:,[2,6]].unique(dim=0,return_inverse=True, sorted=True)
        temp_rel_emb = self.rela_embed(output[:,0])
        hr = temp_rel_emb[reverse_indexes]
        hs = hidden[sub]


        r_idx = edges[:, 0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = self.gate(hr, h_qr, hs)
        # 注意力应该可以换成1层自注意力+1层其他注意力
        alpha = self.w_alpha(self.leakyrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        sigmoid_attention = torch.sigmoid(alpha)
        up_message = sigmoid_attention * message

        message_agg = scatter(up_message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        ones = torch.ones(size=(up_message.shape[0],1),dtype=torch.float32,device=message_agg.device)
        degrees = scatter(ones, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = message_agg/torch.sqrt(degrees+1e-4)
        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new



class TimeEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, max_length=1000):
        super().__init__()
        self.d_model = hidden_dim
        self.max_length = max_length
        
        # 创建嵌入矩阵
        self.embedding = torch.nn.Embedding(max_length, hidden_dim)
        
        # 计算正弦和余弦函数的值
        pos = torch.arange(0, max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        
        # 在嵌入矩阵中初始化编码
        self.embedding.weight.data = pe
        self.embedding.weight.requires_grad = False
        
    def forward(self, x):
        # 将时间值映射为嵌入表示
        x = self.embedding(x)
        return x
    

class GateUnit(nn.Module):
    """
    控制新的信息能够添加到旧有表示
    # todo 替换为使用fc层的模块
    """

    def __init__(self, factor_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_W = nn.Sequential(nn.Linear(self.hidden_size * 2+factor_size, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, factor: torch.Tensor, hidden_state: torch.Tensor)->torch.Tensor:
        """
        通过类似GRU的门控机制更新实体表示

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        factors = torch.cat([message, factor, hidden_state], dim=1)
        # 计算门, 计算门时考虑到查询的关系
        update_value, reset_value = self.gate_W(factors).chunk(2, dim=1)
        # 计算候选隐藏表示
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state

