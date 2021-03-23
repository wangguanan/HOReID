# HOReID
[CVPR2020] High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification. [paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Order_Information_Matters_Learning_Relation_and_Topology_for_Occluded_Person_CVPR_2020_paper.html)

### Update
2020-12: We release a strong baseline for Occluded Re-ID. [link](https://github.com/wangguanan/light-reid/tree/master/examples/occluded_reid)

2020-06-16: Update Code.

2020-04-01: Happy April's Fool Day!!! Code is comming soon.

### Bibtex
If you find the code useful, please consider citing our paper:
```
@InProceedings{wang2020cvpr,
author = {Wang, Guan'an and Yang, Shuo and Liu, Huanyu and Wang, Zhicheng and Yang, Yang and Wang, Shuliang and Yu, Gang and Zhou, Erjin and Sun, Jian},
title = {High-Order Information Matters: Learning Relation and Topology for Occluded Person Re-Identification},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Set Up
```shell script
conda create -n horeid python=3.7
conda activate horeid
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
# GPU Memory >= 10G, Memory >= 20G
```


### Preparation
* Dataset: Occluded DukeMTMC-reID ([Project](https://github.com/lightas/Occluded-DukeMTMC-Dataset))
* Pre-trained Pose Model ([pose_hrnet_w48_256x192.pth](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC), 
please download it to path ```./core/models/model_keypoints/pose_hrnet_w48_256x192.pth```)


### Trained Model 
* [BaiDuDisk](https://pan.baidu.com/s/10TQ221aPz5-FMaW2YP2NJw) (pwd:fgit)
* Google Drive (comming soon)

### Train
```
python main.py --mode train \
--duke_path path/to/occluded/duke \
--output_path ./results 
```

### Test with Trained Model
```
python main.py --mode test \
--resume_test_path path/to/pretrained/model --resume_test_epoch 119 \
--duke_path path/to/occluded/duke --output_path ./results
```

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: guan.wang0706@gmail.com

