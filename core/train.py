import torch
from .models import mining_hard_pairs, analyze_ver_prob
from tools import *


def train_an_epoch(config, base, loaders, epoch):
    base.set_train()
    meter = MultiItemAverageMeter()

    ### we assume 200 iterations as an epoch
    for _ in range(200):
        ### load a batch data
        imgs, pids, _ = loaders.train_iter.next_one()
        imgs, pids = imgs.to(base.device), pids.to(base.device)

        ### forward
        feature_info, cls_score_info, ver_probs, gmp_info, gmn_info, keypoints_confidence = base.forward(imgs, pids, training=True)

        feature_vector_list, gcned_feature_vector_list = feature_info
        cls_score_list, gcned_cls_score_list = cls_score_info
        ver_prob_p, ver_prob_n = ver_probs
        s_p, emb_p, emb_pp = gmp_info
        s_n, emb_n, emb_nn = gmn_info

        ### loss
        ide_loss = base.compute_ide_loss(cls_score_list, pids, keypoints_confidence)
        triplet_loss = base.compute_triplet_loss(feature_vector_list, pids)
        ### gcn loss
        gcned_ide_loss = base.compute_ide_loss(gcned_cls_score_list, pids, keypoints_confidence)
        gcned_triplet_loss = base.compute_triplet_loss(gcned_feature_vector_list, pids)
        ### graph matching loss
        s_gt = torch.eye(14).unsqueeze(0).repeat([s_p.shape[0], 1, 1]).detach().to(base.device)
        pp_loss = base.permutation_loss(s_p, s_gt)
        pn_loss = base.permutation_loss(s_n, s_gt)
        p_loss = pp_loss # + pn_loss
        ### verification loss
        ver_loss = base.bce_loss(ver_prob_p, torch.ones_like(ver_prob_p)) + base.bce_loss(ver_prob_n, torch.zeros_like(ver_prob_n))

        # overall loss
        loss = ide_loss + gcned_ide_loss + triplet_loss + gcned_triplet_loss
        if epoch >= config.use_gm_after:
            loss += \
               config.weight_p_loss * p_loss + \
               config.weight_ver_loss * ver_loss
        acc = base.compute_accuracy(cls_score_list, pids)
        gcned_acc = base.compute_accuracy(gcned_cls_score_list, pids)
        ver_p_ana = analyze_ver_prob(ver_prob_p, True)
        ver_n_ana = analyze_ver_prob(ver_prob_n, False)

        ### optimize
        base.optimizer.zero_grad()
        loss.backward()
        base.optimizer.step()

        ### recored
        meter.update({'ide_loss': ide_loss.data.cpu().numpy(), 'gcned_ide_loss': gcned_ide_loss.data.cpu().numpy(),
                      'triplet_loss': triplet_loss.data.cpu().numpy(), 'gcned_triplet_loss': gcned_triplet_loss.data.cpu().numpy(),
                      'acc': acc, 'gcned_acc': gcned_acc,
                      'ver_loss': ver_loss.data.cpu().numpy(), 'ver_p_ana': torch.tensor(ver_p_ana).data.cpu().numpy(), 'ver_n_ana': torch.tensor(ver_n_ana).data.cpu().numpy(),
                      'pp_loss': pp_loss.data.cpu().numpy(), 'pn_loss': pn_loss.data.cpu().numpy()})

    return meter.get_val(), meter.get_str()



