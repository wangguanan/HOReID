import math
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_adj(node_num, linked_edges, self_connect=1):
    '''
    Params:
        node_num: node number
        linked_edges: [[from_where, to_where], ...]
        self_connect: float,
    '''

    if self_connect > 0:
        adj = np.eye(node_num) * self_connect
    else:
        adj = np.zeros([node_num] * 2)

    for i, j in linked_edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # we suppose the last one is global feature
    adj[-1, :-1] = 0
    adj[-1, -1] = 1
    print(adj)

    adj = torch.from_numpy(adj.astype(np.float32))
    return adj


class AdaptDirGraphGonvLayer(nn.Module):

    def __init__(self, in_dim, out_dim, adj, scale):
        super(AdaptDirGraphGonvLayer, self).__init__()

        # parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.adj = adj
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.reset_parameters()
        self.out = 0

        # layers for adj
        self.fc_direct = nn.Linear(in_dim, 1, bias=False)
        self.bn_direct = nn.BatchNorm1d(in_dim)
        self.sigmoid = nn.Sigmoid()

        # layers for feature
        self.fc_original_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_merged_feature = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):

        # learn adj
        adj2 = self.learn_adj(inputs, self.adj)

        # merge feature
        merged_inputs = torch.matmul(adj2, inputs)
        outputs1 = self.fc_merged_feature(merged_inputs)

        # embed original feature
        outputs2 = self.fc_original_feature(inputs)

        outputs = self.relu(outputs1) + outputs2
        return outputs

    def learn_adj(self, inputs, adj):

        # inputs [bs, k(node_num), c]
        bs, k, c = inputs.shape

        #
        global_features = inputs[:, k - 1, :].unsqueeze(1).repeat([1, k, 1])  # [bs,k,2048]
        distances = torch.abs(inputs - global_features)  # [bs, k, 2048]

        # bottom triangle
        distances_gap = []
        position_list = []
        for i, j in itertools.product(list(range(k)), list(range(k))):
            if i < j and (i != k - 1 and j != k - 1) and adj[i, j] > 0:
                distances_gap.append(distances[:, i, :].unsqueeze(1) - distances[:, j, :].unsqueeze(1))
                position_list.append([i, j])
        distances_gap = 15 * torch.cat(distances_gap, dim=1)  # [bs, edge_number, 2048]
        adj_tmp = self.sigmoid(self.scale * self.fc_direct(
            self.bn_direct(distances_gap.transpose(1, 2)).transpose(1, 2))).squeeze()  # [bs, edge_number]

        # re-assign
        adj2 = torch.ones([bs, k, k]).cuda()
        for indx, (i, j) in enumerate(position_list):
            adj2[:, i, j] = adj_tmp[:, indx] * 2
            adj2[:, j, i] = (1 - adj_tmp[:, indx]) * 2

        mask = adj.unsqueeze(0).repeat([bs, 1, 1])
        new_adj = adj2 * mask
        new_adj = F.normalize(new_adj, p=1, dim=2)

        return new_adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'


class GraphConvNet(nn.Module):

    def __init__(self, adj, in_dim, hidden_dim, out_dim, scale):
        super(GraphConvNet, self).__init__()

        self.adgcn1 = AdaptDirGraphGonvLayer(in_dim, hidden_dim, adj, scale)
        self.adgcn2 = AdaptDirGraphGonvLayer(hidden_dim, out_dim, adj, scale)
        self.relu = nn.ReLU(inplace=True)

    def __call__(self, feature_list):
        cated_features = [feature.unsqueeze(1) for feature in feature_list]
        cated_features = torch.cat(cated_features, dim=1)

        middle_features = self.adgcn1(cated_features)
        out_features = self.adgcn2(middle_features)

        out_feats_list = []
        for i in range(out_features.shape[1]):
            out_feat_i = out_features[:, i].squeeze(1)
            out_feats_list.append(out_feat_i)

        return out_feats_list
