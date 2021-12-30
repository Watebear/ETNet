import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.losses import *

class TNet_lstm(nn.Module):
    def __init__(self, config):
        super(TNet_lstm, self).__init__()
        self.use_gpu = config.use_gpu
        self.training = config.training
        self.au_num = config.au_num

        self.bi = config.bi
        self.num_layers = config.num_layers
        self.unit_dim = config.unit_dim
        self.au_num = config.au_num
        self.feat_dim = config.feat_dim

        self.lstm = nn.LSTM(input_size=self.feat_dim,
                            hidden_size=self.unit_dim * 64,
                            num_layers=self.num_layers,
                            bidirectional=self.bi)
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.unit_dim * 64, self.au_num * 2)

        self.au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        if self.use_gpu:
            self.au_weight = au_weight.float().cuda()
        else:
            self.au_weight = au_weight.float()

    def forward(self, feats, au=None):
        #print('before: ', feats.shape)
        feats = torch.transpose(feats, 0, 1)
        feats = feats.contiguous()
        #print('after: ', feats.shape)

        feats_output, (h_n, c_n) = self.lstm(feats)
        #feats_output = feats_output.transpose(1, 0)
        #print('feats_output_2:', feats_output.shape)
        #feats_output = feats_output.contiguous().view(feats_output.size(0), feats_output.size(1) * feats_output.size(2))
        #print('feats_output3:', feats_output.shape)
        feats_output = self.fc1(feats_output[2])
        feats_output = self.dropout1(feats_output)
        feats_output = self.fc2(feats_output)
        au_output = feats_output.view(feats_output.size(0), 2, int(feats_output.size(1) / 2))
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
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')
    parser.add_argument('--feat_dim', type=int, default=512, help='input dim of lstm')
    parser.add_argument('--num_layers', type=int, default=5, help='num layers of lstm')


    # Directories.
    parser.add_argument('--model_prefix', type=str, default='DISFA_combine_2_3')
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_2_3')
    parser.add_argument('--train_path_prefix', type=str, default='../../data/list/DISFA_combine_2_3')
    parser.add_argument('--test_path_prefix', type=str, default='../../data/list/DISFA_combine_2_3')
    parser.add_argument('--flip_reflect', type=str, default='../../data/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/base_jaa1/')
    config = parser.parse_args()



