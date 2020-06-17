import torch
import numpy as np
from tools import cosine_dist, visualize_ranked_results, time_now, NumpyCatMeter, TorchCatMeter


def visualize_ranked_images(config, base, loaders, dataset):

    base.set_eval()

    # init dataset
    if dataset == 'market':
        _datasets = [loaders.market_query_samples.samples, loaders.market_gallery_samples.samples]
        _loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
        save_visualize_path = base.save_visualize_market_path
    elif dataset == 'duke':
        _datasets = [loaders.duke_query_samples.samples, loaders.duke_gallery_samples.samples]
        _loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
        save_visualize_path = base.save_visualize_duke_path

    # compute featuress
    query_features, query_features2, gallery_features, gallery_features2 = compute_features(base, _loaders, True)

    # compute cosine similarity
    cosine_similarity = cosine_dist(
        torch.tensor(query_features).cuda(),
        torch.tensor(gallery_features).cuda()).data.cpu().numpy()

    # compute verification score
    ver_scores = compute_ver_scores(cosine_similarity, query_features2, gallery_features2, base.verificator, topk=25, sort='descend')

    # visualize
    visualize_ranked_results(cosine_similarity, ver_scores, _datasets, save_dir=save_visualize_path, topk=20, sort='descend')



def compute_features(base, loaders, use_gcn):

    # meters
    query_features_meter, query_features2_meter, query_pids_meter, query_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()
    gallery_features_meter, gallery_features2_meter, gallery_pids_meter, gallery_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()

    # compute query and gallery features
    with torch.no_grad():
        for loader_id, loader in enumerate(loaders):
            for data in loader:
                # compute feautres
                images, pids, cids = data
                images, pids, cids = images.to(base.device), pids.to(base.device), cids.to(base.device)
                info, gcned_info = base.forward(images, pids, training=False)
                features_stage1, features_stage2 = info
                gcned_features_stage1, gcned_features_stage2 = gcned_info
                if use_gcn:
                    features_stage1 = gcned_features_stage1
                    features_stage2 = gcned_features_stage2
                else:
                    features_stage1 = features_stage1
                    features_stage2 = features_stage2

                # save as query features
                if loader_id == 0:
                    query_features_meter.update(features_stage1.data.cpu().numpy())
                    query_features2_meter.update(features_stage2.data.cpu().numpy())
                    query_pids_meter.update(pids.cpu().numpy())
                    query_cids_meter.update(cids.cpu().numpy())
                # save as gallery features
                elif loader_id == 1:
                    gallery_features_meter.update(features_stage1.data.cpu().numpy())
                    gallery_features2_meter.update(features_stage2.data.cpu().numpy())
                    gallery_pids_meter.update(pids.cpu().numpy())
                    gallery_cids_meter.update(cids.cpu().numpy())

    #
    query_features = query_features_meter.get_val()
    query_features2 = query_features2_meter.get_val()
    gallery_features = gallery_features_meter.get_val()
    gallery_features2 = gallery_features2_meter.get_val()

    return query_features, query_features2, gallery_features, gallery_features2


def compute_ver_scores(cosine_similarity, query_features_stage2, gallery_features_stage2, verificator, topk, sort='descend'):
    assert sort in ['ascend', 'descend']
    ver_scores_list = []
    distance_stage1 = cosine_similarity
    #
    for sample_idnex in range(distance_stage1.shape[0]):
        # stage 1, compute distance, return index and topk
        a_sample_distance_stage1 = distance_stage1[sample_idnex]
        if sort == 'descend':
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)[::-1]
        elif sort == 'ascend':
            a_sample_index_stage1 = np.argsort(a_sample_distance_stage1)
        a_sample_topk_index_stage1 = a_sample_index_stage1[:topk]
        # stage2: feature extract topk features
        a_sample_query_feature_stage2 = query_features_stage2[sample_idnex]
        topk_gallery_features_stage2 = gallery_features_stage2[a_sample_topk_index_stage1]
        a_sample_query_feature_stage2 = \
            torch.Tensor(a_sample_query_feature_stage2).cuda().unsqueeze(0).repeat([topk, 1, 1])
        topk_gallery_features_stage2 = torch.Tensor(topk_gallery_features_stage2).cuda()

        # stage2: compute verification score
        with torch.no_grad():
            probs = verificator(a_sample_query_feature_stage2, topk_gallery_features_stage2)
            probs = probs.detach().view([-1]).cpu().data.numpy()

        ver_scores_list.append(np.expand_dims(probs, axis=0))

    ver_scores = np.concatenate(ver_scores_list, axis=0)
    return ver_scores
