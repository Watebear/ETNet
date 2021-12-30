import os
import numpy as np
import glob

data_root = '/home/dddzz/workspace/Datasets/CK+'
img_root = '/home/dddzz/workspace/Datasets/CK+/cohn-kanade-images'
fac_root = '/home/dddzz/workspace/Datasets/CK+/FACS'
ldm_root = '/home/dddzz/workspace/Datasets/CK+/Landmarks'

im_list = []
for sub in os.listdir(img_root):
    for seq in os.listdir(os.path.join(img_root, sub)):
        if '.DS_Store' in seq:
            continue
        seq_list = []
        for fac in os.listdir(os.path.join(img_root, sub, seq)):
            if '.DS_Store' in fac:
                continue
            seq_list.append(os.path.join(sub, seq, fac))
        # adjust the order
        seq_list.sort(key=lambda x: int(x[-6:-4]))
        im_list += seq_list

ck_im_list = open('ck_im_list.txt', 'w')
for line in im_list:
    ck_im_list.write(line+'\n')
ck_im_list.close()








