import numpy as np
import os

class TensorList(object):
    def __init__(self,
                 path,
                 tensor_path,
                 num_frames):
        super().__init__()
        self.num_frames = num_frames
        assert (self.num_frames - 1) % 2 == 0
        self.feats = np.load(os.path.join(tensor_path, 'feats.npy'))
        self.labels = np.load(os.path.join(tensor_path, 'labels.npy'))
        self.image_list = open(path + '_path.txt').readlines()
        self.tuples = self.make_tuples()

    def make_tuples(self):
        r = int((self.num_frames - 1) // 2)
        tuples = []
        for i in range(len(self.image_list)):
            im_id = int(self.image_list[i].strip().split('/')[2][:-4])
            try:
                if im_id - r != int(self.image_list[i-r].strip().split('/')[2][:-4]):
                    continue
                if im_id + r != int(self.image_list[i+r].strip().split('/')[2][:-4]):
                    continue
                tuples.append([int(j) for j in range(i-r, i+r+1)])
            except:
                continue
        return tuples

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, index):
        ids = self.tuples[index]
        start = ids[0]
        end = ids[-1]
        m = int((self.num_frames - 1) // 2)
        mid = ids[m]

        feats = self.feats[start:end+1]
        aus = self.labels[mid]

        return feats, aus


class TensorSingle(object):
    def __init__(self, tensor_path):
        super().__init__()
        self.feats = np.load(os.path.join(tensor_path, 'feats.npy'))
        self.labels = np.load(os.path.join(tensor_path, 'labels.npy'))
        #self.feats, self.labels = self.clean()

    def __len__(self):
        return len(self.feats)

    def clean(self):
        new_feats = []
        new_labels = []
        # !!!
        for i in range(len(self.feats)):
            feat = self.feats[i]
            label = self.labels[i]
            if sum(label) < 1:
                continue
            else:
                new_feats.append(feat)
                new_labels.append(label)
        return new_feats, new_labels

    def __getitem__(self, index):
        feats = self.feats[index]
        aus = self.labels[index]
        return feats, aus


if __name__ == "__main__":
    '''
    ds = TensorList(
        path='../data/list/DISFA_combine_2_3',
        tensor_path='../data/tensor/DISFA_combine_2_3',
        num_frames=5,
    )
    feats, _ = ds.__getitem__(0)
    print(feats.shape)
    '''
    ds = TensorList(
        path='./list/DISFA_combine_2_3',
        tensor_path='tensor_512/DISFA_combine_2_3/train',
        num_frames=3
    )
    feats, _ = ds.__getitem__(0)
    print(feats.shape)