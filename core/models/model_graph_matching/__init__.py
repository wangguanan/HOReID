import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import Sinkhorn
from .voting_layer import Voting
from .gconv import Siamese_Gconv
from .affinity_layer import Affinity
from .permutation_loss import PermutationLoss

from tools import cosine_dist, label2similarity


class GMNet(nn.Module):
    def __init__(self):
        super(GMNet, self).__init__()

        self.BS_ITER_NUM = 20
        self.BS_EPSILON = 1e-10
        self.FEATURE_CHANNEL = 2048
        self.GNN_FEAT = 1024
        self.GNN_LAYER = 2
        self.VOT_ALPHA = 200.0

        self.bi_stochastic = Sinkhorn(max_iter=self.BS_ITER_NUM, epsilon=self.BS_EPSILON)
        self.voting_layer = Voting(self.VOT_ALPHA)

        for i in range(self.GNN_LAYER):
            gnn_layer = Siamese_Gconv(self.FEATURE_CHANNEL, self.GNN_FEAT) if i == 0 else Siamese_Gconv(self.GNN_FEAT, self.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(self.GNN_FEAT))
            if i == self.GNN_LAYER - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(self.GNN_FEAT * 2, self.GNN_FEAT))


    def forward(self, emb1_list, emb2_list, adj):

        if type(emb1_list).__name__ == type(emb2_list).__name__ == 'list':
            emb1 = torch.cat([emb1.unsqueeze(1) for emb1 in emb1_list], dim=1)
            emb2 = torch.cat([emb2.unsqueeze(1) for emb2 in emb2_list], dim=1)
        else:
            emb1 = emb1_list
            emb2 = emb2_list

        org_emb1 = emb1
        org_emb2 = emb2

        ns_src = (torch.ones([emb1.shape[0]]) * 14).int()
        ns_tgt = (torch.ones([emb2.shape[0]]) * 14).int()

        for i in range(self.GNN_LAYER):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([adj, emb1], [adj, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)
            s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)

            if i == self.GNN_LAYER - 2:
                emb1_before_cross, emb2_before_cross = emb1, emb2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1_before_cross, torch.bmm(s, emb2_before_cross)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2_before_cross, torch.bmm(s.transpose(1, 2), emb1_before_cross)), dim=-1))

        fin_emb1 = org_emb1 + torch.bmm(s, org_emb2)
        fin_emb2 = org_emb2 + torch.bmm(s.transpose(1,2), org_emb1)

        return s, fin_emb1, fin_emb2



class Verificator(nn.Module):

    def __init__(self, config):
        super(Verificator, self).__init__()

        self.config = config

        self.bn = nn.BatchNorm1d(2048*14)
        self.layer1 = nn.Linear(2048*14, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def __call__(self, feature_vectors_list1, feature_vectors_list2):
        '''
        :param feature_vectors_list1: list with length node_num, element size is [bs, feature_length]
        :param feature_vectors_list2: list with length node_num, element size is [bs, feature_length]
        :return:
        '''


        if type(feature_vectors_list1).__name__ == 'list':
            feature_vectors_1 = torch.cat(
                [feature_vector1.unsqueeze(1) for feature_vector1 in feature_vectors_list1], dim=1)
            feature_vectors_2 = torch.cat(
                [feature_vector2.unsqueeze(1) for feature_vector2 in feature_vectors_list2], dim=1)
        elif type(feature_vectors_list1).__name__ == 'Tensor': # [bs, branch_num, channel_num]
            feature_vectors_1 = feature_vectors_list1
            feature_vectors_2 = feature_vectors_list2

        # feature_vectors_1 = feature_vectors_1.detach()
        # feature_vectors_2 = feature_vectors_2.detach()

        feature_vectors_1 = F.normalize(feature_vectors_1, p=2, dim=2)
        feature_vectors_2 = F.normalize(feature_vectors_2, p=2, dim=2)

        features = self.config.ver_in_scale * feature_vectors_1 * feature_vectors_2
        features = features.view([features.shape[0], features.shape[1] * features.shape[2]])

        logit = self.layer1(features)
        prob = self.sigmoid(logit)

        return prob


def mining_hard_pairs(feature_vector_list, pids):
    '''
    use global feature (the last one) to mining hard positive and negative pairs
    cosine distance is used to measure similarity
    :param feature_vector_list:
    :param pids:
    :return:
    '''

    global_feature_vectors = feature_vector_list[-1]
    dist_matrix = cosine_dist(global_feature_vectors, global_feature_vectors)
    label_matrix = label2similarity(pids, pids).float()

    _, sorted_mat_distance_index = torch.sort(dist_matrix + (9999999.) * (1 - label_matrix), dim=1, descending=False)
    hard_p_index = sorted_mat_distance_index[:, 0]
    _, sorted_mat_distance_index = torch.sort(dist_matrix + (-9999999.) * (label_matrix), dim=1, descending=True)
    hard_n_index = sorted_mat_distance_index[:, 0]

    new_feature_vector_list = []
    p_feature_vector_list = []
    n_feature_vector_list = []
    for feature_vector in feature_vector_list:
        feature_vector = copy.copy(feature_vector)
        new_feature_vector_list.append(feature_vector.detach())
        feature_vector = copy.copy(feature_vector.detach())
        p_feature_vector_list.append(feature_vector[hard_p_index, :])
        feature_vector = copy.copy(feature_vector.detach())
        n_feature_vector_list.append(feature_vector[hard_n_index, :])

    return new_feature_vector_list, p_feature_vector_list, n_feature_vector_list


def analyze_ver_prob(prob, positive):
    '''
    :param prob: [bs, 1]
    :param positive: True or False
    :return:
    '''

    if positive:
        hit = (prob > 0.5).float()
        unhit = (prob < 0.5).float()
    else:
        hit = (prob < 0.5).float()
        unhit = (prob > 0.5).float()

    avg_prob = torch.mean(prob)
    acc = torch.mean(hit)
    avg_hit_prob = torch.sum(prob * hit) / torch.sum(hit)
    avg_unhit_prob = torch.sum(prob * unhit) / torch.sum(unhit)

    return avg_prob, acc, avg_hit_prob, avg_unhit_prob
