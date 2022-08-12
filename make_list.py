import os
from glob import glob

def make_list():
    cities = os.listdir('./data/cityscapes/leftImg8bit/test/')

    with open('data/list/cityscapes/test.lst', 'w') as f:
        for city in cities:
            img_names = os.listdir('./data/cityscapes/leftImg8bit/test/' + city)
            for name in img_names:
                f.write('leftImg8bit/test/' + city + '/' + name + '\t' + 'gtFine/test/' + city + '/' +
                        name.split('leftImg8bit')[0] + 'gtFine_labelIds.png' + '\n')

make_list()