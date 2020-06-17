import torch
import numpy as np
from tools import time_now, NumpyCatMeter, TorchCatMeter, CMC, CMCWithVer


def testwithVer2(config, logger, base, loaders, test_dataset, use_gcn, use_gm):
    base.set_eval()

    # meters
    query_features_meter, query_features2_meter, query_pids_meter, query_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()
    gallery_features_meter, gallery_features2_meter, gallery_pids_meter, gallery_cids_meter = NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter(), NumpyCatMeter()

    # init dataset
    if test_dataset == 'market':
        loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
    elif test_dataset == 'duke':
        loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]

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

    # compute mAP and rank@k
    query_info = (query_features, query_features2, query_cids_meter.get_val(), query_pids_meter.get_val())
    gallery_info = (gallery_features, gallery_features2, gallery_cids_meter.get_val(), gallery_pids_meter.get_val())

    alpha = 0.1 if use_gm else 1.0
    topk = 8
    mAP, cmc = CMCWithVer()(query_info, gallery_info, base.verificator, base.gmnet, topk, alpha)

    return mAP, cmc
