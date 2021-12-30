import numpy as np
import random
from PIL import Image
import torch

im_root = './data/imgs/'

def make_dataset(image_list, land, biocular, au):
    len_ = len(image_list)
    images = [(image_list[i].strip(), land[i, :], biocular[i], au[i, :]) for i in range(len_)]
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageList_5(object):
    def __init__(self,
                 crop_size,
                 path,
                 phase='train',
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        image_list = open(path + '_path.txt').readlines()
        land = np.loadtxt(path + '_land.txt')
        biocular = np.loadtxt(path + '_biocular.txt')
        au = np.loadtxt(path + '_AUoccur.txt')
        imgs = make_dataset(image_list, land, biocular, au)
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase

        self.tuples = self.make_tuples()

    def make_tuples(self):
        tuples = []
        for i in range(len(self.imgs)):
            im_id = self.imgs[i][0].strip().split('/')[2][:-4]
            try:
                im_id_next_1 = self.imgs[i + 1][0].strip().split('/')[2][:-4]
                im_id_next_2 = self.imgs[i + 2][0].strip().split('/')[2][:-4]
            except:
                continue
            if im_id == '0' or im_id == '1' or im_id_next_1 == '0' or im_id_next_2 == '0':
                continue
            else:
                tuples.append([i-1, i-2, i, i+1, i+2])
        return tuples

    def __getitem__(self, index):
        ids = self.tuples[index]
        imgs, lands, bioculars, aus = [], [], [], []
        for id in ids:
            path, land, biocular, au = self.imgs[id]
            au = np.array([1. if a > 0 else 0. for a in au])
            path = im_root + path
            img = self.loader(path)
            if self.phase == 'train':
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)

                flip = random.randint(0, 1)

                if self.transform is not None:
                    img = self.transform(img, flip, offset_x, offset_y)
                if self.target_transform is not None:
                    land = self.target_transform(land, flip, offset_x, offset_y)
            # for testing
            else:
                w, h = img.size
                offset_y = (h - self.crop_size) / 2
                offset_x = (w - self.crop_size) / 2
                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is not None:
                    land = self.target_transform(land, 0, offset_x, offset_y)
            imgs.append(img.unsqueeze(dim=0))
            lands.append(land)
            bioculars.append(biocular)
            aus.append(au)

        imgs = torch.cat(imgs, dim=0)  # torch.Size([5, 3, 176, 176])

        return imgs, lands[2], bioculars[2], aus[2]

    def __len__(self):
        return len(self.tuples)


if __name__ == "__main__":
    crop_size = 176
    path = '../data/list/DISFA_combine_1_2'
    ds = ImageList_5(crop_size, path)
    imgs, lands, bioculars, aus = ds.__getitem__(0)
    for im in imgs:
        print(im.size)

