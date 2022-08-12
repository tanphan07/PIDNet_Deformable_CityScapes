import os

import torch
# import numpy as np
# from torchsummary import summary
# from models.pidnet import PIDNet, get_pred_model
# from DCNv2.dcn_v2 import DCN

# model = PIDNet(m=2, n=3, num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True)
#
# dummy_input = torch.rand(1, 3, 1024, 2048)
# #
# # model.cuda()
# #
# dummy_input.cuda()
# #
# # summary(model, dummy_input)
#
# dcn = DCN(3, 512, kernel_size=(3,3), padding=1, stride=1, deformable_groups=1).cuda()
# out = dcn(dummy_input)
# print(out.shape)
# input = torch.randn(2, 64, 128, 128).cuda()
#     # wrap all things (offset and mask) in DCN
# dcn = DCN(64, 512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1).cuda()
# output = dcn(input)
# print(output.shape)
city = os.listdir('./data/cityscapes/gtFine/train')
k= 0
for ci in city:
    m = os.listdir('./data/cityscapes/gtFine/train/' + ci)
    k += len(m)
print(k)