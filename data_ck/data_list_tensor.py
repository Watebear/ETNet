import numpy as np
import os

feat_root = '/home/dddzz/workspace/Codes/Knightly/Peace/data_ck/feats'

class Tensor_List(object):
    def __init__(self, samples):
        self.samples = samples
        self.tensors, self.labels = self.load_samples()

    # actual: 1, 2, 6, 7, 12, 15, 17, 23, 24
    def load_samples(self):
        tensors = []
        labels = []
        for line in self.samples:
            line = line.strip().split()
            # tensor
            path_tgt = line[0].replace('.png', '.npy')
            id_tgt = int(path_tgt[-12:-4])
            id_4 = "0"*(8-len(str(id_tgt - 1))) + str(id_tgt - 1)
            id_3 = "0"*(8-len(str(id_tgt - 2))) + str(id_tgt - 2)
            id_2 = "0"*(8-len(str(id_tgt - 3))) + str(id_tgt - 3)
            id_1 = "0"*(8-len(str(id_tgt - 4))) + str(id_tgt - 4)
            path_tgt = os.path.join(feat_root, path_tgt)
            path_4 = path_tgt.replace(path_tgt[-12:-4], id_4)
            path_3 = path_tgt.replace(path_tgt[-12:-4], id_3)
            path_2 = path_tgt.replace(path_tgt[-12:-4], id_2)
            path_1 = path_tgt.replace(path_tgt[-12:-4], id_1)
            tensor_tgt = np.expand_dims(np.load(path_tgt), axis=0)
            tensor_4 = np.expand_dims(np.load(path_4), axis=0)
            tensor_3 = np.expand_dims(np.load(path_3), axis=0)
            tensor_2 = np.expand_dims(np.load(path_2), axis=0)
            tensor_1 = np.expand_dims(np.load(path_1), axis=0)
            tensor = np.concatenate([tensor_1, tensor_2, tensor_3, tensor_4, tensor_tgt])
            tensors.append(tensor)
            # label
            au1, au2, au6, au12 = int(line[1]), int(line[2]), int(line[3]), int(line[5])
            label = np.array([au1, au2, au6, au12])
            labels.append(label)
        return tensors, labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        feats = self.tensors[index]
        aus = self.labels[index]
        return feats, aus

if __name__ == "__main__":
    samples = open('all_au_list.txt', 'r').readlines()

    ds = Tensor_List(samples)
