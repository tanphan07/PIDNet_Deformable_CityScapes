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
from torch.nn import functional as F
from tqdm import tqdm
from models.pidnet import PIDNet, get_pred_model
from datasets.cityscapes import Cityscapes
from configs import config
from configs import update_config
import matplotlib.pyplot as plt
from utils.utils import decode_segmap
from PIL import Image

color_map = [(128, 64, 128),
             (244, 35, 232),
             (70, 70, 70),
             (102, 102, 156),
             (190, 153, 153),
             (153, 153, 153),
             (250, 170, 30),
             (220, 220, 0),
             (107, 142, 35),
             (152, 251, 152),
             (70, 130, 180),
             (220, 20, 60),
             (255, 0, 0),
             (0, 0, 142),
             (0, 0, 70),
             (0, 60, 100),
             (0, 80, 100),
             (0, 0, 230),
             (119, 11, 32)]


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


def main(image, anno, stt, load_pt=False):
    model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # check_point_path = './output/pidnet_small_cityscapes/best.pt'
    if load_pt:
        # load checkpoint for .pt model
        check_point_path = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/DCNV2_600epoch/best.pt'
        checkpoint = torch.load(check_point_path, map_location={'cuda:0': 'cpu'})
        model.load_state_dict(checkpoint)
    else:
        # load checkpoint for .pth.tar model
        model_state_file = '/home/tanpv/workspace/SelfDC/PIDNet/output/pidnet_small_cityscapes/checkpoint_480.pth.tar'
        checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
        dct = checkpoint['state_dict']
        model.load_state_dict(
            {k.replace('', ''): v for k, v in dct.items() if k.startswith('')})

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    size = anno.size()
    out_put = model(image)[1]
    # print(out_put.shape)
    out_put = F.interpolate(
        input=out_put, size=size[-2:],
        mode='bilinear', align_corners=True
    )
    out_put = torch.argmax(out_put, dim=1)

    for jj in range(image.size()[0]):
        img = image.numpy()
        show_output = out_put.numpy()
        gt = anno.numpy()
        # print(np.unique(gt))
        tmp = np.array(show_output[jj]).astype(np.uint8)

        tmp1 = np.array(gt[jj]).astype(np.uint8)
        # print(np.unique(tmp1))
        # segmap = decode_segmap(tmp, dataset='cityscapes')
        #
        # segmap1 = decode_segmap(tmp1, dataset='cityscapes')

        ignore_index = np.where(tmp1 == 255)
        tmp[ignore_index] = 255
        sv_img = np.zeros((1024, 2048, 3))
        for i, color in enumerate(color_map):
            for j in range(3):
                sv_img[:, :, j][tmp == i] = color_map[i][j]
        anno_img = np.zeros((1024, 2048, 3))

        for i, color in enumerate(color_map):
            for j in range(3):
                anno_img[:, :, j][tmp1 == i] = color_map[i][j]
        # print(np.unique(segmap1))
        img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
        img_tmp *= (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        # cv2.imwrite('Image.png', img_tmp)
        # cv2.imwrite('Annotaion.png', anno_img)
        # cv2.imwrite('Prediction.png', sv_img)
        # h_img = cv2.vconcat([anno_img, sv_img])
        cv2.imwrite('/home/tanpv/workspace/SelfDC/PIDNet/ouput_image/anno/anno_' + (stt) + '_.png', anno_img)
        # plt.figure()
        # plt.title('display')
        # plt.subplot(311)
        # plt.imshow(img_tmp)
        # plt.subplot(312)
        # plt.imshow(segmap)
        # plt.subplot(313)
        # plt.imshow(segmap1)
    # plt.show(block=True)


if __name__ == '__main__':
    test_dataset = Cityscapes(
        root='data/',
        list_path='list/cityscapes/test.lst',
        num_classes=19,
        multi_scale=False,
        flip=False,
        ignore_label=255,
        base_size=2048,
        crop_size=(1024, 2048)
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    for i, (test_img, anno, _, _, name) in tqdm(enumerate(testloader), total=len(testloader)):
        main(test_img, anno, name[0])
