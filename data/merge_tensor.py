import numpy as np
import os

pa = './tensor_12000/DISFA_combine_1_2'
if not os.path.exists(pa):
    os.mkdir(pa)

feat1 = './tensor_12000/DISFA_part1/feats.npy'
feat1 = np.load(feat1)
feat2 = './tensor_12000/DISFA_part2/feats.npy'
feat2 = np.load(feat2)
feat3 = np.concatenate([feat1, feat2], axis=0)
np.save(os.path.join(pa, 'feats.npy'), feat3)


label1 = './tensor_12000/DISFA_part1/labels.npy'
label1 = np.load(label1)
label2 = './tensor_12000/DISFA_part2/labels.npy'
label2 = np.load(label2)
label3 = np.concatenate([label1, label2], axis=0)
np.save(os.path.join(pa, 'labels.npy'), label3)

print(feat1.shape)
print(feat2.shape)
print(feat3.shape)
print(label1.shape)
print(label2.shape)
print(label3.shape)

