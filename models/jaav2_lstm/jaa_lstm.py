import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.jaav2_lstm.network import HMRegionLearning, AlignNet, LocalAttentionRefine, LocalAUNetv2, HLFeatExtractor, AUNet
from models.losses.losses import au_softmax_loss, au_dice_loss, landmark_loss
from models.jaav2_lstm.t_net import T_net_lstm

class JaaNetv2_lstm(nn.Module):
    def __init__(self, config):
        super(JaaNetv2_lstm, self).__init__()
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

        self.t_net = T_net_lstm(feat_dim=512, unit_dim=config.unit_dim, au_num=8, num_layers=5)

        # params
        self.init_models()
        if config.freeze_jaa:
            self.freeze_jaanet()

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
                       self.local_au_net, self.global_au_feat]:
            for param in subnet.parameters():
                param.requires_grad = False

    def get_jaafeat(self, inputs):
        align_feat_list = []
        align_output_list = []
        local_au_out_feat_list = []
        local_aus_output_list = []
        global_au_out_feat_list = []
        concat_au_feat_list = []

        for i in range(5):
            input = inputs[:, i, ...]
            region_feat = self.region_learning(input)
            align_feat, align_output, aus_map = self.align_net(region_feat)
            if self.use_gpu:
                aus_map = aus_map.cuda()
            output_aus_map = self.local_attention_refine(aus_map.detach())
            local_au_out_feat, local_aus_output = self.local_au_net(region_feat, output_aus_map)
            global_au_out_feat = self.global_au_feat(region_feat)
            concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
            concat_au_feat = self.au_net(concat_au_feat)

            align_feat_list.append(align_feat)
            align_output_list.append(align_output)
            local_au_out_feat_list.append(local_au_out_feat)
            local_aus_output_list.append(local_aus_output)
            global_au_out_feat_list.append(global_au_out_feat)
            concat_au_feat_list.append(concat_au_feat.unsqueeze(0))

        return align_feat_list, align_output_list, local_au_out_feat_list, local_aus_output_list, global_au_out_feat_list, concat_au_feat_list

    # only use t_net merge concat feat
    def forward(self, input, au=None, land=None, biocular=None):
        align_feat_list, align_output_list, local_au_out_feat_list, local_aus_output_list,\
        global_au_out_feat_list, concat_au_feat_list = self.get_jaafeat(input)

        local_aus_output = local_aus_output_list[2]
        align_output = align_output_list[2]
        aus_output = self.t_net(concat_au_feat_list)

        if self.training:
            loss_au_softmax = au_softmax_loss(aus_output, au, weight=self.au_weight)
            loss_au_dice = au_dice_loss(aus_output, au, weight=self.au_weight)
            loss_au = loss_au_softmax + loss_au_dice

            loss_local_au_softmax = au_softmax_loss(local_aus_output, au, weight=self.au_weight)
            loss_local_au_dice = au_dice_loss(local_aus_output, au, weight=self.au_weight)
            loss_local_au = loss_local_au_softmax + loss_local_au_dice

            loss_land = landmark_loss(align_output, land, biocular)
            total_loss = self.lambda_au * (loss_au + loss_local_au) + self.lambda_land * loss_land
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
    # Model configuration.
    parser.add_argument('--freeze_jaa', type=bool, default=True, help='freeze jaanet params')
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
    config.training = False

    net = JaaNetv2_lstm(config).cuda()
    inp = torch.randn((10, 5, 3, 176, 176)).cuda()
    aus_output, local_aus_output, align_output = net(inp)
    torch.save(net.state_dict(), './demo.pth')



