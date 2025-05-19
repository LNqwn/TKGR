import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torchdrug import layers
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

import numpy as np

class TimeEncode(torch.nn.Module):
    '''
    This class refer to the Bochner's time embedding
    time_dim: int, dimension of temporal entity embeddings
    relation_specific: bool, whether use relation specific freuency and phase.
    num_relations: number of relations.
    '''

    def __init__(self, time_dim, relation_specific=False, num_relations=None):

        super(TimeEncode, self).__init__()
        self.time_dim = time_dim
        self.relation_specific = relation_specific
###########################
        if relation_specific:  # shape: num_relations * time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_relations, 1))
            self.phase = torch.nn.Parameter(
                torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_relations, 1))
        else:  # shape: time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())
        # #####################################
        if relation_specific:   # shape: num_relations * time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_relations, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_relations, 1))
        else:  # shape: time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, relations=None):
        '''
        :param ts: [edge_num, seq_len]
        :param relations: which relations do we extract their time embeddings.
        :return: [edge_num, seq_len, time_dim]
        '''
        edge_num = ts.size(0)
        seq_len = ts.size(1) # seq_len = 1
        ts = torch.unsqueeze(ts, dim=2)
        
        if self.relation_specific:
            # self.basis_freq[relations]:  [edge_num, time_dim]
            map_ts = ts * self.basis_freq[relations].unsqueeze(dim=1)  # [edge_num, 1, time_dim]
            map_ts += self.phase[relations].unsqueeze(dim=1)
        else:
            # self.basis_freq:  [time_dim]
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [edge_num, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic

class TemporalPathAgg(MessagePassing):
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", time_encoding=True, time_encoding_independent=True):
        super(TemporalPathAgg, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.time_encoding = time_encoding
        self.time_encoding_independent = time_encoding_independent

        if time_encoding:
            self.time_encoder = TimeEncode(time_dim=self.input_dim, relation_specific=time_encoding_independent,
                                           num_relations=num_relation)
            self.relation4time = nn.Sequential(
                nn.Linear(input_dim * 2 , input_dim),
                nn.ReLU()
            )

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)

#edge_index 的第一行包含每条边的起始节点索引，第二行包含每条边的目标节点索引
    def forward(self, input, query, initial_stat, edge_index, edge_type, edge_time, size, edge_weight=None):
        batch_size = len(query)
        # layer-specific relation features as a projection of query r embeddings
        relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        output = self.propagate(input=input, relation=relation, initial_stat=initial_stat, edge_index=edge_index,
                                edge_type=edge_type, edge_time=edge_time, size=size, edge_weight=edge_weight)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        return super(TemporalPathAgg, self).propagate(edge_index, size, **kwargs)

    def message(self, input_j, relation, initial_stat, edge_type, edge_time):
        
        relation_emb = relation.index_select(self.node_dim, edge_type)
        # time encoding
        if self.time_encoding:
            if self.time_encoding_independent:#形状为 (edge_num, seq_len)，表示每条边的时间序列。edge_type仅在 relation_specific=True 时需要。形状为 (edge_num,)，表示每条边对应的关系
                time_emb = self.time_encoder((edge_time-edge_time.min()).unsqueeze(1), edge_type)
            else:
                time_emb = self.time_encoder((edge_time-edge_time.min()).unsqueeze(1))     
            time_emb = torch.squeeze(time_emb, 1)
            relation_j = self.relation4time(torch.cat([relation_emb, time_emb.repeat(relation_emb.shape[0],1,1)], dim=-1))
        else:
            relation_j = relation_emb

        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the initial_stat
        message = torch.cat([message, initial_stat], dim=self.node_dim)  # (batch_size, num_edges + num_nodes, input_dim)

        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        index = torch.cat([index, torch.arange(dim_size, device=input.device)])
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            eps = 1e-6
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)

        return output

    def update(self, update, input):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
###########################################################
class TemporalPathAggStarGraph(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", time_encoding=True, time_encoding_independent=True):
        super(TemporalPathAggStarGraph, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        #self.dependent = dependent
        self.time_encoding = time_encoding
        self.time_encoding_independent = time_encoding_independent
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        # if dependent:
        #     self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        # else:
        #     self.relation = nn.Embedding(num_relation * 2, input_dim)

        # force rspmm to be compiled, to avoid cold start in time benchmark
        adjacency = torch.rand(1, 1, 1, device=self.device).to_sparse()
        relation_input = torch.rand(1, 32, device=self.device)
        input = torch.rand(1, 32, device=self.device)
        functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul="mul")

    def compute_message(self, node_input, edge_input , edge_time):

        #relation_emb = relation.index_select(self.node_dim, edge_type)
        # time encoding
        if self.time_encoding:
            if self.time_encoding_independent:
                time_emb = self.time_encoder((edge_time - edge_time.min()).unsqueeze(1), edge_type)
            else:
                time_emb = self.time_encoder((edge_time - edge_time.min()).unsqueeze(1))
            time_emb = torch.squeeze(time_emb, 1)
            edge_input = self.relation4time(
                torch.cat([relation_emb, time_emb.repeat(relation_emb.shape[0], 1, 1)], dim=-1))

        if self.message_func == "transe":
            return node_input + edge_input
        elif self.message_func == "distmult":
            return node_input * edge_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            return torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

    def compute_message_noTime(self, node_input, edge_input):
        if self.message_func == "transe":
            return node_input + edge_input
        elif self.message_func == "distmult":
            return node_input * edge_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            return torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
    def message(self, graph, input):
        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        # if self.dependent:
        #     relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        # else:
        #     relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
        else:
            relation_input = relation_input.transpose(0, 1)

        node_input = input[node_in]
        edge_input = relation_input[relation]
        edge_time = graph.edata['time']
        message = self.compute_message(node_input, edge_input, edge_time)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        batch_size = len(graph.query)
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = getattr(graph, "pna_degree_out", graph.degree_out)
        degree_out = degree_out.unsqueeze(-1) + 1
        if not isinstance(graph, data.PackedGraph):
            edge_weight = edge_weight.unsqueeze(-1)
            degree_out = degree_out.unsqueeze(-1)

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        if not isinstance(graph, data.PackedGraph):
            update = update.view(len(update), batch_size, -1)
        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation * 2

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)
        node_in, node_out, relation = graph.edge_list.t()

        degree_out = getattr(graph, "pna_degree_out", graph.degree_out)
        degree_out = degree_out.unsqueeze(-1) + 1
        relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        # if self.dependent:
        #     relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        # else:
        #     relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), graph.edge_weight,
                                                (graph.num_node, graph.num_node, batch_size * graph.num_relation))
        else:
            relation_input = relation_input.transpose(0, 1).flatten(1)
            adjacency = graph.adjacency
        adjacency = adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        if not isinstance(graph, data.PackedGraph):
            update = update.view(len(update), batch_size, -1)
        return update

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


