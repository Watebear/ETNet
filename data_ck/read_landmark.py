import os
import cv2
import numpy as np

data_root = '/home/dddzz/workspace/Datasets/CK+'
img_root = '/home/dddzz/workspace/Datasets/CK+/cohn-kanade-images'
fac_root = '/home/dddzz/workspace/Datasets/CK+/FACS'
ldm_root = '/home/dddzz/workspace/Datasets/CK+/Landmarks'

ck_list_txt = 'all_au_list.txt'
ck_list = open(ck_list_txt, 'r')
lines = ck_list.readlines()

for line in lines:
    line = line.strip().split()
    im_path = os.path.join(img_root, line[0])
    ld_path = os.path.join(ldm_root, line[0].replace('.png', '_landmarks.txt'))
    img = cv2.imread(im_path)

    ldms = np.loadtxt(ld_path)
    for i in range(len(ldms)):
        p = ldms[i]
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (int(p[0]), int(p[1])), font,  0.5, (0, 0, 255), 1)
    cv2.imwrite('ck_ldm_demo.png', img)

    l1 = ldms[61]
    l2 = ldms[62]
    l3 = ldms[63]
    l4 = ldms[65]
    l5 = ldms[66]
    l6 = ldms[67]

    ldms[60] = l1
    ldms[61] = l2
    ldms[62] = l3
    ldms[63] = l4
    ldms[64] = l5
    ldms[65] = l6
    ldms = ldms[17:66]

    print(len(ldms))

    img = cv2.imread(im_path)
    for i in range(len(ldms)):
        p = ldms[i]
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (int(p[0]), int(p[1])), font,  0.3, (0, 0, 255), 1)
    cv2.imwrite('ck_ldm_demo_1.png', img)

    #cv2.imshow('a', img)
    #cv2.waitKey()



    [][1]


