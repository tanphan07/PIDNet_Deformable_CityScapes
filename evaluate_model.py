# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import time
import timeit
import cv2
import numpy as np
from torchvision import transforms as tf
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from models.pidnet import PIDNet, get_pred_model
from datasets.cityscapes import Cityscapes
from configs import config
from configs import update_config
from utils.function import testval1


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main(pt_model=False):
    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # check_point_path = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/checkpoint_480.pth.tar'
    # check_point_path = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/DCNV1_600epoch/best.pt'
    if pt_model:
        check_point_path = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/DCNV2_600epoch/best.pt'
        checkpoint = torch.load(check_point_path, map_location={'cuda:0': 'cpu'})

        model.load_state_dict(checkpoint)

    else:
        model_state_file = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/checkpoint_480.pth.tar'
        checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
        dct = checkpoint['state_dict']
        model.load_state_dict(
            {k.replace('', ''): v for k, v in dct.items() if k.startswith('')})

    model = model.to(torch.device('cuda'))
    test_size = (1024, 2048)
    test_dataset = Cityscapes(
        root='data/',
        list_path='list/cityscapes/test.lst',
        num_classes=19,
        multi_scale=False,
        flip=False,
        ignore_label=255,
        base_size=2048,
        crop_size=test_size)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    testval1(testloader=testloader, model=model)


if __name__ == '__main__':
    main()
