import os
import time
import torch


def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))


def label2similarity(label1, label2):
    '''
    compute similarity matrix of label1 and label2
    :param label1: torch.Tensor, [m]
    :param label2: torch.Tensor, [n]
    :return: torch.Tensor, [m, n], {0, 1}
    '''
    m, n = len(label1), len(label2)
    l1 = label1.view(m, 1).expand([m, n])
    l2 = label2.view(n, 1).expand([n, m]).t()
    similarity = l1 == l2
    return similarity