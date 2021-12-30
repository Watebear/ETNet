import cv2
import numpy as np
import os
import math

data_root = '/home/dddzz/workspace/Datasets/CK+'
img_root = '/home/dddzz/workspace/Datasets/CK+/cohn-kanade-images'
fac_root = '/home/dddzz/workspace/Datasets/CK+/FACS'
ldm_root = '/home/dddzz/workspace/Datasets/CK+/Landmarks'

# params
box_enlarge = 2.9
img_size = 200


def visualize(img, pts):
    pts = np.array(pts).reshape((49, 2))
    img_new = img.copy()
    for i in range(49):
        p = pts[i]
        cv2.circle(img_new, (int(p[0]), int(p[1])), 2, (255, 255, 0), 1)
    cv2.imshow('tmp', img_new)
    cv2.waitKey(0)

def align_face_49pts(img, img_land, box_enlarge, img_size):
    leftEye0 = (img_land[2 * 19] + img_land[2 * 20] + img_land[2 * 21] + img_land[2 * 22] + img_land[2 * 23] +
                img_land[2 * 24]) / 6.0
    leftEye1 = (img_land[2 * 19 + 1] + img_land[2 * 20 + 1] + img_land[2 * 21 + 1] + img_land[2 * 22 + 1] +
                img_land[2 * 23 + 1] + img_land[2 * 24 + 1]) / 6.0
    rightEye0 = (img_land[2 * 25] + img_land[2 * 26] + img_land[2 * 27] + img_land[2 * 28] + img_land[2 * 29] +
                 img_land[2 * 30]) / 6.0
    rightEye1 = (img_land[2 * 25 + 1] + img_land[2 * 26 + 1] + img_land[2 * 27 + 1] + img_land[2 * 28 + 1] +
                 img_land[2 * 29 + 1] + img_land[2 * 30 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 13], img_land[2 * 13 + 1], 1],
                   [img_land[2 * 31], img_land[2 * 31 + 1], 1], [img_land[2 * 37], img_land[2 * 37 + 1], 1]])

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land

def ck_plus_68_to_49(ldms):
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
    return ldms

def visualize(img, pts):
    pts = np.array(pts).reshape((49, 2))
    img_new = img.copy()
    for i in range(49):
        p = pts[i]
        cv2.circle(img_new, (int(p[0]), int(p[1])), 2, (255, 255, 0), 1)
    cv2.imshow('tmp', img_new)
    cv2.waitKey(0)

def array_to_str(lands):
    land_list = []
    for la in lands:
        land_list.append(str(la))
    land_str = " ".join(land_list)
    return land_str

def process_dataset():
    new_im_root = './imgs'
    new_ld_root = './ldms'

    im_list_file = open('ck_im_list.txt', 'r')
    im_list = im_list_file.readlines()
    for im_path in im_list:
        im_path = im_path.strip()
        # load landmarks
        lm_path = im_path.replace(".png", "_landmarks.txt")
        lm_path_1 = os.path.join(ldm_root, lm_path)
        lands = np.loadtxt(lm_path_1)
        lands = ck_plus_68_to_49(lands).reshape(-1)
        # load images
        im_path_1 = os.path.join(img_root, im_path)
        img = cv2.imread(im_path_1)
        # align
        aligned_img, new_land = align_face_49pts(img, lands, box_enlarge, img_size)
        # save imgs
        save_im_path = os.path.join(new_im_root, im_path)
        cv2.imwrite(save_im_path, aligned_img)
        # save landmarks
        new_land_str = array_to_str(new_land)
        save_lm_path = os.path.join(new_ld_root, lm_path)
        new_land_txt = open(save_lm_path, 'w')
        new_land_txt.write(new_land_str+'\n')
        new_land_txt.close()


if __name__ == "__main__":
    process_dataset()












