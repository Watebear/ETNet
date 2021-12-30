import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math

from models.jaav2_multi.transformer import TransformerEncoder
from models.losses.losses import au_softmax_loss, au_dice_loss

class T_net_multi_7(nn.Module):
    def __init__(self, configs,):
        """
        Construct a MulT model.
        """
        super(T_net_multi_7, self).__init__()
        # setting
        self.ori_dim = 512
        self.emb_dim = configs.emb_dim          # emb dim of transformer
        self.comb_emb_dim = self.emb_dim * 7  # cat all
        self.num_heads = configs.num_heads
        self.num_layers = configs.num_layers
        self.attn_mask = configs.attn_mask
        self.au_num = configs.au_num

        # dropout
        self.attn_dropout = configs.attn_dropout
        self.relu_dropout = configs.relu_dropout
        self.embed_dropout = configs.embed_dropout
        self.res_dropout = configs.res_dropout
        self.out_dropout = configs.out_dropout

        # 1. Temporal convolutional layers , may not useful here
        self.proj_1 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_2 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_3 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_4 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_5 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_6 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        self.proj_7 = nn.Linear(self.ori_dim, self.emb_dim, bias=False)
        # 2. Cross Attentions
        self.trans_cross = self.get_network(self.emb_dim)
        # 3. Self Attention
        self.trans_self = self.get_network(self.emb_dim)
        # 4. Merge
        self.trans_mem = self.get_network(self.comb_emb_dim)
        # 5. Residual
        self.proj1 = nn.Linear(self.comb_emb_dim, self.comb_emb_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.out_dropout)
        self.proj2 = nn.Linear(self.comb_emb_dim, self.comb_emb_dim)
        # 6. Output
        self.out_layer = nn.Linear(self.comb_emb_dim, self.au_num * 2)

        init_prior = 0.01
        for m in [self.proj1, self.proj2]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))

        # model
        self.training = configs.training
        self.use_gpu = configs.use_gpu
        self.au_weight = torch.from_numpy(np.loadtxt(configs.train_path_prefix + '_weight.txt'))
        au_weight = torch.from_numpy(np.loadtxt(configs.train_path_prefix + '_weight.txt'))
        if self.use_gpu:
            self.au_weight = au_weight.float().cuda()
        else:
            self.au_weight = au_weight.float()

    def get_network(self, emb_dim):
        return TransformerEncoder(embed_dim=emb_dim,
                                  num_heads=self.num_heads,
                                  layers=self.num_layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, feats, au=None):
        if self.ori_dim != self.emb_dim:
            feat1 = self.proj_1(feats[:, 0, :]).unsqueeze(0)
            feat2 = self.proj_1(feats[:, 1, :]).unsqueeze(0)
            feat3 = self.proj_1(feats[:, 2, :]).unsqueeze(0)
            feat4 = self.proj_1(feats[:, 3, :]).unsqueeze(0)
            feat5 = self.proj_1(feats[:, 4, :]).unsqueeze(0)
            feat6 = self.proj_1(feats[:, 5, :]).unsqueeze(0)
            feat7 = self.proj_1(feats[:, 6, :]).unsqueeze(0)
        else:
            feat1 = feats[:, 0, :].unsqueeze(0)
            feat2 = feats[:, 1, :].unsqueeze(0)
            feat3 = feats[:, 2, :].unsqueeze(0)
            feat4 = feats[:, 3, :].unsqueeze(0)
            feat5 = feats[:, 4, :].unsqueeze(0)
            feat6 = feats[:, 5, :].unsqueeze(0)
            feat7 = feats[:, 6, :].unsqueeze(0)

        # cross attention
        feat_1_4 = self.trans_cross(feat4, feat1, feat1)
        feat_2_4 = self.trans_cross(feat4, feat2, feat2)
        feat_3_4 = self.trans_cross(feat4, feat4, feat4)
        feat_5_4 = self.trans_cross(feat4, feat5, feat5)
        feat_6_4 = self.trans_cross(feat4, feat6, feat6)
        feat_7_4 = self.trans_cross(feat4, feat7, feat7)
        # self attention
        feat_4_4 = self.trans_self(feat4, feat4, feat4)
        # merge
        feat_merge = torch.cat([feat_1_4, feat_2_4, feat_3_4, feat_4_4, feat_5_4, feat_6_4, feat_7_4], dim=2)
        feat_merge = self.trans_mem(feat_merge).squeeze(0)
        # residual
        feat_merge_proj = self.proj1(self.dropout1(self.relu1(self.proj1(feat_merge))))
        feat_out = feat_merge + feat_merge_proj
        # output
        au_output = self.out_layer(feat_out)
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1) / 2))
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
    # training configuration
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    # T_net configuration
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--proj_dim', type=int, default=16,
                        help='target frame embedding dim of transformer layer (default: 512)')
    parser.add_argument('--attn_dropout', type=float, default=0.2,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.2,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.2,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.2,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.5,
                        help='output layer dropout')
    parser.add_argument('--attn_mask', default=True, action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    # path configuration
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_1_2')
    parser.add_argument('--train_path_prefix', type=str, default='../../data/list/DISFA_combine_1_2')

    configs = parser.parse_args()
    configs.training = False

    inp = torch.randn(5, 1, 512)
    net = T_net_multi(configs)
    oup = net(inp)
    print(oup.shape)

