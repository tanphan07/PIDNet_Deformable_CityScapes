from datasets.cityscapes import Cityscapes
import torch
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import string

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
             (119, 11, 32), (0, 0, 0)]

normalize_colormap = [tuple(i / 255 for i in j) for j in color_map]

# print(normalize_colormap)
if __name__ == '__main__':
    phases = ['test', 'val', 'train']

    for phase in phases:
        print('==========' + phase + '============')
        test_dataset = Cityscapes(
            root='data/',
            list_path='list/cityscapes/' + phase + '.lst',
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

        label = {'road': 0,
                 'sidewalk': 1,
                 'building': 2,
                 'wall': 3,
                 'fence': 4,
                 'pole': 5,
                 'traffic light': 6,
                 'traffic sign': 7,
                 'vegetation': 8,
                 'terrain': 9,
                 'sky': 10,
                 'person': 11,
                 'rider': 12,
                 'car': 13,
                 'truck': 14,
                 'bus': 15,
                 'train': 16,
                 'motorcycle': 17,
                 'bicycle': 18,
                 'ignore_label': 19
                 }
        label_array = tuple(label.keys())
        # print(label_array)
        result = np.zeros(20, dtype=np.uint64)
        print(result)
        i = 0
        for img, anno, _, _, _ in tqdm(testloader, total=len(testloader)):
            ignore_index = np.where(anno == 255)
            anno[ignore_index] = 19
            values, counts = np.unique(anno, return_counts=True)

            result[values] += counts.astype(np.uint64)
            # i += 1
            # if i == 50:
            #     break
        print(result)
        plt.yscale('symlog')
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        # color = ["#" + ''.join(random.choices("ABCDEF" + string.digits, k=6) for ]
        num_max = np.amax(result) * 1.2
        num_min = np.amin(result) * 0.8
        plt.ylim([num_min, num_max])
        plt.ylabel('Số lượng pixel')
        y_pos = np.arange(len(label_array))
        plt.bar(y_pos, result, color=normalize_colormap)
        plt.xticks(y_pos, label_array, rotation=90)
        plt.subplots_adjust(bottom=0.3, top=0.98)
        plt.savefig(phase + '_histogram.png')
