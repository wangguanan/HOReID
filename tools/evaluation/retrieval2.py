import numpy as np
from sklearn import metrics as sk_metrics

import torch
import torch.nn.functional as F


class CMC:
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''

    def __init__(self):
        pass

    def __call__(self, query_info, gallery_info, dist):

        query_feature, query_cam, query_label = query_info
        gallery_feature, gallery_cam, gallery_label = gallery_info
        assert dist in ['cosine', 'euclidean']
        print(query_feature.shape, gallery_feature.shape)

        if dist == 'cosine':
            # distance = self.cosine_dist_torch(
            #     torch.Tensor(query_feature).cuda(),
            #     torch.Tensor(gallery_feature).cuda()).data.cpu().numpy()
            distance = self.cosine_dist(query_feature, gallery_feature)
        elif dist == 'euclidean':
            # distance = self.euclidean_dist_torch(
            #     torch.Tensor(query_feature).cuda(),
            #     torch.Tensor(gallery_feature).cuda()).data.cpu().numpy()
            distance = self.euclidean_dist(query_feature, gallery_feature)

        APs = []
        CMC = []
        query_num = query_feature.shape[0]
        for i in range(query_num):
            AP, cmc = self.evaluate(
                distance[i],
                query_cam[i], query_label[i],
                gallery_cam, gallery_label, dist)
            APs.append(AP)
            CMC.append(cmc)

        mAP = np.mean(np.array(APs))

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        CMC = np.mean(np.array(CMC), axis=0)

        return mAP, CMC


    def evaluate(self, distance, query_cam, query_label, gallery_cam, gallery_label, dist):

        if dist is 'cosine':
            index = np.argsort(distance)[::-1]
        elif dist is 'euclidean':
            index = np.argsort(distance)

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)


    def compute_AP(self, index, good_index):
        '''
        :param index: np.array, 1d
        :param good_index: np.array, 1d
        :return:
        '''

        num_good = len(good_index)
        hit = np.in1d(index, good_index)
        index_hit = np.argwhere(hit == True).flatten()

        if len(index_hit) == 0:
            AP = 0
            cmc = np.zeros([len(index)])
        else:
            precision = []
            for i in range(num_good):
                precision.append(float(i+1) / float((index_hit[i]+1)))
            AP = np.mean(np.array(precision))
            cmc = np.zeros([len(index)])
            cmc[index_hit[0]: ] = 1

        return AP, cmc


    def in1d(self, array1, array2, invert=False):
        '''
        :param set1: np.array, 1d
        :param set2: np.array, 1d
        :return:
        '''
        mask = np.in1d(array1, array2, invert=invert)
        return array1[mask]


    def notin1d(self, array1, array2):
        return self.in1d(array1, array2, invert=True)


    def cosine_dist_torch(self, x, y):
        '''
        :param x: torch.tensor, 2d
        :param y: torch.tensor, 2d
        :return:
        '''
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return torch.mm(x, y.transpose(0, 1))


    def euclidean_dist_torch(self, mtx1, mtx2):
        """
        mtx1 is an autograd.Variable with shape of (n,d)
        mtx1 is an autograd.Variable with shape of (n,d)
        return a nxn distance matrix dist
        dist[i,j] represent the L2 distance between mtx1[i] and mtx2[j]
        """
        m = mtx1.size(0)
        p = mtx1.size(1)
        mmtx1 = torch.stack([mtx1] * m)
        mmtx2 = torch.stack([mtx2] * m).transpose(0, 1)
        dist = torch.sum((mmtx1 - mmtx2) ** 2, 2).squeeze()
        return dist


    def cosine_dist(self, x, y):
        return 1 - sk_metrics.pairwise.cosine_distances(x, y)


    def euclidean_dist(self, x, y):
        return sk_metrics.pairwise.euclidean_distances(x, y)




class CMCWithVer(CMC):
    '''
    Compute Rank@k and mean Average Precision (mAP) scores
    Used for Person ReID
    Test on MarKet and Duke
    '''


    def __call__(self, query_info, gallery_info, verificator, gmnet, topk, alpha):
        '''
        use cosine + verfication loss as distance
        '''

        query_features_stage1, query_features_stage2, query_cam, query_label = query_info
        gallery_features_stage1, gallery_features_stage2, gallery_cam, gallery_label = gallery_info

        APs = []
        CMC = []

        # compute distance
        # distance_stage1 = self.cosine_dist_torch(
        #     torch.Tensor(query_features_stage1).cuda(),
        #     torch.Tensor(gallery_features_stage1).cuda()).data.cpu().numpy()
        distance_stage1 = self.cosine_dist(query_features_stage1, gallery_features_stage1)

        #
        for sample_idnex in range(distance_stage1.shape[0]):
            a_sample_query_cam = query_cam[sample_idnex]
            a_sample_query_label = query_label[sample_idnex]

            # stage 1, compute distance, return index and topk
            a_sample_distance_stage1 = distance_stage1[sample_idnex]
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)[::-1]
            a_sample_topk_index_stage1 = a_sample_index_stage1[:topk]

            # stage2: feature extract topk features
            a_sample_query_feature_stage2 = query_features_stage2[sample_idnex]
            topk_gallery_features_stage2 = gallery_features_stage2[a_sample_topk_index_stage1]
            a_sample_query_feature_stage2 = \
                torch.Tensor(a_sample_query_feature_stage2).cuda().unsqueeze(0).repeat([topk, 1, 1])
            topk_gallery_features_stage2 = torch.Tensor(topk_gallery_features_stage2).cuda()

            # stage2: compute verification score
            with torch.no_grad():
                _, a_sample_query_feature_stage2, topk_gallery_features_stage2 = \
                    gmnet(a_sample_query_feature_stage2, topk_gallery_features_stage2, None)
                probs = verificator(a_sample_query_feature_stage2, topk_gallery_features_stage2)
                probs = probs.detach().view([-1]).cpu().data.numpy()

            # stage2 index
            # print(a_sample_distance_stage1[a_sample_topk_index_stage1])
            # print(probs)
            # print(1-probs)
            # print('*******')
            topk_distance_stage2 = alpha * a_sample_distance_stage1[a_sample_topk_index_stage1] + (1 - alpha) * (1-probs)
            topk_index_stage2 = np.argsort(topk_distance_stage2)[::-1]
            topk_index_stage2 = a_sample_topk_index_stage1[topk_index_stage2.tolist()]
            a_sample_index_stage2 = np.concatenate([topk_index_stage2, a_sample_index_stage1[topk:]])

            #
            ap, cmc = self.evaluate(
                a_sample_index_stage2, a_sample_query_cam, a_sample_query_label, gallery_cam, gallery_label, 'cosine')
            APs.append(ap)
            CMC.append(cmc)

        mAP = np.mean(np.array(APs))

        min_len = 99999999
        for cmc in CMC:
            if len(cmc) < min_len:
                min_len = len(cmc)
        for i, cmc in enumerate(CMC):
            CMC[i] = cmc[0: min_len]
        CMC = np.mean(np.array(CMC), axis=0)

        return mAP, CMC



    def evaluate(self, index, query_cam, query_label, gallery_cam, gallery_label, dist):

        junk_index_1 = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam == gallery_cam))
        junk_index_2 = np.argwhere(gallery_label == -1)
        junk_index = np.append(junk_index_1, junk_index_2)

        good_index = self.in1d(np.argwhere(query_label == gallery_label), np.argwhere(query_cam != gallery_cam))
        index_wo_junk = self.notin1d(index, junk_index)

        return self.compute_AP(index_wo_junk, good_index)