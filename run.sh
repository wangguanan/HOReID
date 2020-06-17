#!/usr/bin/env bash

#python main.py --duke_path /home/wangguanan/datasets/PersonReIDDatasets/Duke/occlude_DukeMTMC-reID/ --output_path ./results --mode train

python main.py \
--mode test \
--resume_test_path /data/projects/20190703_Partial-ReID/13.0-HO-ReID/2.3.1.1.1.1.1.3.1.1.1.1-bot+keypoints+gcn+gm-norm10/out/gm_lr_scale_1.0/weight_p_loss/1.0/models --resume_test_epoch 119 \
--duke_path /home/wangguanan/datasets/PersonReIDDatasets/Duke/occlude_DukeMTMC-reID/ --output_path ./results
