import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import *
from models.losses.losses import au_softmax_loss, au_dice_loss, landmark_loss
from models.jaav2_base.network import HMRegionLearning, AlignNet, LocalAttentionRefine, LocalAUNetv2, HLFeatExtractor, AUNet

class JaaNetv2(nn.Module):
    def __init__(self, config):
        super(JaaNetv2, self).__init__()
        # misc
        self.training = config.training
        self.use_gpu = config.use_gpu
        self.config = config

        # model
        self.region_learning = HMRegionLearning(input_dim=3, unit_dim=config.unit_dim)
        self.align_net = AlignNet(crop_size=config.crop_size, map_size=config.map_size,
                                  au_num=config.au_num, land_num=config.land_num,
                                  input_dim=config.unit_dim * 8)
        self.local_attention_refine = LocalAttentionRefine(au_num=config.au_num, unit_dim=config.unit_dim)
        self.local_au_net = LocalAUNetv2(au_num=config.au_num, input_dim=config.unit_dim * 8,
                                         unit_dim=config.unit_dim)
        self.global_au_feat = HLFeatExtractor(input_dim=config.unit_dim * 8, unit_dim=config.unit_dim)
        self.au_net = AUNet(au_num=config.au_num, input_dim=12000, unit_dim=config.unit_dim)

        # params
        self.init_models()

        self.au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        if self.use_gpu:
            self.au_weight = au_weight.float().cuda()
        else:
            self.au_weight = au_weight.float()

        self.lambda_au = config.lambda_au
        self.lambda_land = config.lambda_land

    def init_models(self):
        self.region_learning.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/region_learning.pth'))
        self.align_net.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/align_net.pth'))
        self.local_attention_refine.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/local_attention_refine.pth'))
        self.local_au_net.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/local_au_net.pth'))
        self.global_au_feat.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/global_au_feat.pth'))
        self.au_net.load_state_dict(torch.load(
            self.config.pretrain_prefix + '/au_net.pth'), strict=False)

    def freeze_jaanet(self):
        for subnet in [self.region_learning, self.align_net, self.local_attention_refine,
                       self.local_au_net, self.global_au_feat, self.au_net]:
            for param in subnet.parameters():
                param.requires_grad = False

    def forward(self, input, au=None, land=None, biocular=None):
        #print('input:', input.shape)
        region_feat = self.region_learning(input)
        #print('region_feat:', region_feat.shape)
        align_feat, align_output, aus_map = self.align_net(region_feat)
        #print('align_feat:', align_feat.shape)
        #print('align_output:', align_output.shape)
        #print('aus_map:', aus_map.shape)

        if self.use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = self.local_attention_refine(aus_map.detach())
        #print('output_aus_map:', output_aus_map.shape)
        local_au_out_feat, local_aus_output = self.local_au_net(region_feat, output_aus_map)
        #print('local_au_out_feat:', local_au_out_feat.shape)
        #print('local_aus_output:', local_aus_output.shape)
        global_au_out_feat = self.global_au_feat(region_feat)
        #print('global_au_out_feat:', global_au_out_feat.shape)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
        #print('concat_au_feat:', concat_au_feat.shape)
        aus_output = self.au_net(concat_au_feat)
        #print('aus_output:', aus_output.shape)

        if self.training:
            loss_au_softmax = au_softmax_loss(aus_output, au, weight=self.au_weight)
            loss_au_dice = au_dice_loss(aus_output, au, weight=self.au_weight)
            loss_au = loss_au_softmax + loss_au_dice

            loss_local_au_softmax = au_softmax_loss(local_aus_output, au, weight=self.au_weight)
            loss_local_au_dice = au_dice_loss(local_aus_output, au, weight=self.au_weight)
            loss_local_au = loss_local_au_softmax + loss_local_au_dice

            loss_land = landmark_loss(align_output, land, biocular)
            total_loss = self.lambda_au * (loss_au + loss_local_au) + \
                         self.lambda_land * loss_land
            return total_loss, loss_au_softmax, loss_au_dice, loss_local_au_softmax, loss_local_au_dice, loss_land
        else:
            local_aus_output = (local_aus_output[:, 1, :]).exp()
            aus_output = (aus_output[:, 1, :]).exp()
            return aus_output, local_aus_output, align_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training configuration
    parser.add_argument('--training', type=bool, default=False, help='training or testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='default use gpu')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
    parser.add_argument('--use_nesterov', type=str2bool, default=True)
    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=8, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')
    # path configuration
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_1_2')
    parser.add_argument('--train_path_prefix', type=str, default='../../data/list/DISFA_combine_1_2')

    config = parser.parse_args()

    net = JaaNetv2(config).cuda()
    inp = torch.randn((10, 3, 176, 176)).cuda()
    aus_output, local_aus_output, align_output = net(inp)

    print(aus_output.shape)
    print(local_aus_output.shape)
    print(align_output.shape)

    torch.save(net.state_dict(), './demo.pth')


