import cv2
import numpy as np
import os
import math
import glob

new_im_root = './imgs'
new_ld_root = './ldms'

def array_to_str(lands):
    land_list = []
    for la in lands:
        land_list.append(str(la))
    land_str = " ".join(land_list)
    return land_str

def str2int(inp_str):
    if inp_str[-1] == '1':
        oup = int(inp_str[0] + inp_str[2])
    elif inp_str[-1] == '0':
        oup = int(inp_str[0])
    return oup

def visualize(img, pts):
    pts = np.array(pts).reshape((49, 2))
    img_new = img.copy()
    for i in range(49):
        p = pts[i]
        cv2.circle(img_new, (int(p[0]), int(p[1])), 2, (255, 255, 0), 1)
    cv2.imshow('tmp', img_new)
    cv2.waitKey(0)

def fuck_240_bastards():
    im_txt = open('./CK_path.txt', 'w')
    au_txt = open('./CK_AUocur.txt', 'w')
    ld_txt = open('./CK_land.txt', 'w')

    keep = {'1': 0, '2': 1, '6': 2, '7': 3, '12': 4, '15': 5, '17': 6, '23': 7, '24': 8}
    
    facs_root = '/home/dddzz/workspace/Datasets/CK+/FACS'
    land_root = './ldms'
    imgs_root = './imgs'
   
    subs = glob.glob(facs_root+"/*")
    for sub in subs:
        seqs = glob.glob(sub+"/*")
        for seq in seqs:
            txts = glob.glob(seq+"/*.txt")
            for txt_path in txts:
                # au labels
                txt_path = txt_path.strip()
                txt = open(txt_path, 'r')
                facs = txt.readlines()
                au_labels = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
                for fac in facs:
                    fac = fac.strip().split()[0]
                    fac = str2int(fac)
                    if str(fac) in keep:
                        au_labels[keep[str(fac)]] = '1'
                au_to_write = " ".join(au_labels) + '\n'

                # landmarks
                land_path = txt_path.replace(facs_root, land_root).replace("_facs.txt", "_landmarks.txt")
                lands = np.loadtxt(land_path)
                str_lands = array_to_str(lands)
                try:
                    assert len(str_lands.split()) == 98
                except:
                    print(land_path)
                ld_to_write = str_lands + '\n'

                # img paths
                img_path = txt_path.replace(facs_root, imgs_root).replace("_facs.txt", ".png")
                img = cv2.imread(img_path)
                #visualize(img, lands)
                img_to_write = img_path[2:] + '\n'

                # write
                im_txt.write(img_to_write)
                au_txt.write(au_to_write)
                ld_txt.write(ld_to_write)

    im_txt.close()
    au_txt.close()
    ld_txt.close()

def fuck_240_all():
    input_land = np.loadtxt('./CK_land.txt')
    biocular = np.zeros(input_land.shape[0])
    l_ocular_x = np.mean(input_land[:, np.arange(2 * 20 - 2, 2 * 25, 2)], 1)
    l_ocular_y = np.mean(input_land[:, np.arange(2 * 20 - 1, 2 * 25, 2)], 1)
    r_ocular_x = np.mean(input_land[:, np.arange(2 * 26 - 2, 2 * 31, 2)], 1)
    r_ocular_y = np.mean(input_land[:, np.arange(2 * 26 - 1, 2 * 31, 2)], 1)
    biocular = (l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2
    np.savetxt('CK_biocular.txt', biocular, fmt='%f', delimiter='\t')

def fuck_mingyue():
    imgs_AUoccur = np.loadtxt('CK_AUocur.txt')
    AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))
    for i in range(imgs_AUoccur.shape[1]):
        AUoccur_rate[0, i] = sum(imgs_AUoccur[:, i] > 0) / float(imgs_AUoccur.shape[0])
    AU_weight = 1.0 / AUoccur_rate
    AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
    np.savetxt('CK_weight.txt', AU_weight, fmt='%f', delimiter='\t')

if __name__ == "__main__":
    fuck_240_bastards()
    fuck_240_all()
    fuck_mingyue()