import numpy as np
import os
import cv2

im_txt = './list/DISFA_combine_1_2_path.txt'
ld_txt = './list/DISFA_combine_1_2_land.txt'

path_lines = open(im_txt, 'r').readlines()
ldm_lines = np.loadtxt(ld_txt)
landmarks = ldm_lines[0].reshape(49, 2)

print(landmarks)

path = path_lines[0]

#im_path = os.path.join('/home/dddzz/workspace/Codes/Knightly/Peace/data/imgs', path)
#(im_path)


img = cv2.imread('0.jpg')
img = cv2.resize(img, (400, 400))
for i in range(len(landmarks)-3):
    p = landmarks[i] * 2
    cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 255), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(i), (int(p[0]), int(p[1])), font, 0.4, (0, 0, 255), 1)

cv2.imwrite('disfa_lands_demo.png', img)
cv2.imshow('a', img)
cv2.waitKey(0)




