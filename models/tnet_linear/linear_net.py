import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.losses import *

class TNet_Linear(nn.Module):
    def __init__(self, config):
        super(TNet_Linear, self).__init__()
        self.use_gpu = config.use_gpu
        self.training = config.training
        self.au_num = config.au_num

        #self.dropout0 = nn.Dropout(p=0.5)
        #self.proj0 = nn.Linear(512*5, 512)
        #self.relu0 = nn.ReLU()

        self.proj1 = nn.Linear(512*7, 512*7)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.proj2 = nn.Linear(512*7, 512*7)
        # 6. Output
        self.out_layer = nn.Linear(512*7, self.au_num * 2)

        self.au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        if self.use_gpu:
            self.au_weight = au_weight.float().cuda()
        else:
            self.au_weight = au_weight.float()


    def forward(self, feats, au=None):
        feat1 = feats[:, 0, :]
        feat2 = feats[:, 1, :]
        feat3 = feats[:, 2, :]
        feat4 = feats[:, 3, :]
        feat5 = feats[:, 4, :]
        feat6 = feats[:, 5, :]
        feat7 = feats[:, 6, :]
        feats = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7], dim=1)

        feats1 = self.proj1(self.dropout1(self.relu1(self.proj1(feats))))
        feat_out = feats + feats1
        feat_out = self.out_layer(feat_out)
        au_output = feat_out.view(feat_out.size(0), 2, int(feat_out.size(1) / 2))
        au_output = F.log_softmax(au_output, dim=1)

        if self.training:
            loss_au_softmax = au_softmax_loss(au_output, au, weight=self.au_weight)
            loss_au_dice = au_dice_loss(au_output, au, weight=self.au_weight)
            return loss_au_softmax, loss_au_dice
        else:
            au_output = (au_output[:, 1, :]).exp()
            return au_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Misc
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--eval_batch_size', type=int, default=80, help='mini-batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8)
    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')
    # Directories.
    parser.add_argument('--model_prefix', type=str, default='DISFA_combine_1_2')
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_1_2')
    parser.add_argument('--train_path_prefix', type=str, default='../../data/list/DISFA_combine_1_2')
    parser.add_argument('--test_path_prefix', type=str, default='../../data/list/DISFA_combine_1_2')
    parser.add_argument('--flip_reflect', type=str, default='../../data/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/base_jaa1/')
    config = parser.parse_args()

    net = TNet_Linear(config)
    inp = torch.randn((8, 5, 512))
    oup = net(inp)
    print(oup.shape)