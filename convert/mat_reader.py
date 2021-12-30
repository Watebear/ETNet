# -*- coding: utf-8 -*-
import scipy.io
import os
import numpy as np
#import h5py

list_path_prefix = 'D:\project_jaanet\PyTorch-JAANet-master\data\list'

all_imgs_path = open(os.path.join(list_path_prefix ,'DISFA_part3_path.txt')).readlines()

#all_imgs_land = np.loadtxt('dataset\DISFA_combine_1_2_66land.txt')

write_path = 'dataset'



src_path = 'D:\project_jaanet\DISFA\Landmark_Points'
all_imgs_land_mat = {}

for filepath,dirnames,filenames in os.walk(src_path):
    for filename in filenames:
        #print(filepath[-18:-13])
        frame = int(filename[-11:-7])
        sn = filepath[-18:-13]
        if sn in ['SN010','SN008','SN029','SN025','SN023','SN026','SN027','SN032','SN030','SN009','SN028','SN031','SN021','SN024']:
            frame = frame-1
        path = os.path.join(filepath,filename)
        #DISFA/SN002/0.jpg
        key = 'DISFA/'+sn+'/'+str(frame)+'.jpg'
        #print(key)
        land_point = scipy.io.loadmat(path)
        all_imgs_land_mat[key] = land_point['pts']

all_imgs_new_land = np.zeros([len(all_imgs_path),132])
for i in range(len(all_imgs_path)):
    full_path = all_imgs_path[i].strip()
    try:
        new_land = all_imgs_land_mat[full_path]
    except:
        print('error')
        print(full_path)
    all_imgs_new_land[i, :] = new_land.flatten()

np.savetxt(write_path+'\DISFA_part3_66land.txt', all_imgs_new_land, fmt='%f', delimiter='\t')

