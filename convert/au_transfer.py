import os
import numpy as np

list_path_prefix = 'D:\project_jaanet\PyTorch-JAANet-master\data\list'

'''
example of content in 'BP4D_combine_1_2_AUoccur.txt':
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
'''


all_imgs_au = {}
src_path = 'D:\project_jaanet\DISFA\Landmark_Points'


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
        #land_point = scipy.io.loadmat(path)
        au = [0,0,0,0,0,0,0,0]
        all_imgs_au[key] = au

#DISFA on 8 AUs (1, 2, 4, 6, 9, 12, 25, and 26).

au_dict = {}
au_dict['_au1']=0
au_dict['_au2']=1
au_dict['_au4']=2
#au_dict['_au5']=3
au_dict['_au6']=3
au_dict['_au9']=4
au_dict['au12']=5
#au_dict['au15']=7
#au_dict['au17']=8
#au_dict['au20']=9
au_dict['au25']=6
au_dict['au26']=7


panic_count = 0

au_src_path = 'D:\project_jaanet\DISFA\ActionUnit_Labels'
for filepath,dirnames,filenames in os.walk(au_src_path):
    for filename in filenames:
        #print(filename)
        sn = filename[0:5]
        au_key = filename[-8:-4]
        #print(au_key)
        if au_key not in au_dict.keys():
            continue
        path = os.path.join(filepath,filename)
        au_x = open(path).readlines()
        for au_frame in au_x:
            #print(au_frame)
            frame = int(au_frame[:-3])-1
            #print(frame)
            assert frame<4900
            
            au_value = int(au_frame[-2])
            assert au_value in [0,1,2,3,4,5]
            key = 'DISFA/'+sn+'/'+str(frame)+'.jpg'
            #print(key)
            if au_value <2:
                occ = 0
            else:
                occ = 1
                panic_count+=1
            all_imgs_au[key][au_dict[au_key]] = occ
        
        
all_imgs_path = open(os.path.join(list_path_prefix ,'DISFA_part3_path.txt')).readlines()
write_path = 'dataset'
all_imgs_new_au = np.zeros([len(all_imgs_path),8])
for i in range(len(all_imgs_path)):
    full_path = all_imgs_path[i].strip()
    try:
        new_au = all_imgs_au[full_path]
    except:
        print('error')
        print(full_path)
    all_imgs_new_au[i, :] = new_au

panic_count2 = np.sum(all_imgs_new_au)
print(panic_count,panic_count2)
np.savetxt(write_path+'\DISFA_part3_AUoccur.txt', all_imgs_new_au, fmt='%d', delimiter=' ')
        
        
