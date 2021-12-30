import numpy as np
import random
from PIL import Image

im_root = '/home/dddzz/workspace/Codes/Knightly/Peace/data_ck/imgs/'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageList(object):
    def __init__(self,
                 crop_size,
                 path,
                 phase='train',
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        image_list = open('/home/dddzz/workspace/Codes/Knightly/Peace/data_ck/all_im_list.txt', 'r')
        imgs = image_list.readlines()
        image_list.close()
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase

    def __getitem__(self, index):
        path = self.imgs[index].strip()
        path1 = im_root + path
        img = self.loader(path1)
        if self.phase == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)

            flip = random.randint(0, 1)

            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
            #if self.target_transform is not None:
                #land = self.target_transform(land, flip, offset_x, offset_y)
        # for testing
        else:
            w, h = img.size
            offset_y = (h - self.crop_size) / 2
            offset_x = (w - self.crop_size) / 2
            if self.transform is not None:
                img = self.transform(img)
            #if self.target_transform is not None:
                #land = self.target_transform(land, 0, offset_x, offset_y)

        return img, path#, land, biocular, au

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    crop_size = 176
    root_path = '/home/dddzz/workspace/Codes/Knightly/Peace/data_ck/imgs/'
    ds = ImageList(crop_size, root_path)
    tmp = ds.__getitem__(33)
    for tt in tmp:
        print(tt)

