import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.losses import *

class AUNet(nn.Module):
    def __init__(self, config):
        super(AUNet, self).__init__()
        self.training = config.training
        self.use_gpu = config.use_gpu

        self.au_output = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(12000, config.unit_dim * 64),
            nn.Dropout(p=0.2),
            nn.Linear(config.unit_dim * 64, config.au_num * 2)
        )

        #self.au_net.load_state_dict(torch.load(
            #self.config.pretrain_prefix + '/au_net.pth'), strict=False)

        '''
        ckpt = torch.load(config.pretrain_prefix + '/au_net.pth')
        if not self.training:
            self.au_output.weight = torch.nn.Parameter(ckpt['au_output.1.weight'].cuda())
            self.au_output.bias = torch.nn.Parameter(ckpt['au_output.1.bias'].cuda())
        else:
            init_prior = 0.01
            for m in [self.before]:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # to keep the torch random state
                    m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                    torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))
        '''

        init_prior = 0.5
        for m in self.au_output:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))

        self.au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
        if self.use_gpu:
            self.au_weight = au_weight.float().cuda()
        else:
            self.au_weight = au_weight.float()

    def forward(self, x, au=None):
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #x = self.fc1(x)
        au_output = self.au_output(x)
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
    parser.add_argument('--model_prefix', type=str, default='DISFA_combine_2_3')
    parser.add_argument('--pretrain_prefix', type=str, default='../weights/DISFA_combine_2_3')
    parser.add_argument('--train_path_prefix', type=str, default='data1/list/DISFA_combine_2_3')
    parser.add_argument('--test_path_prefix', type=str, default='data1/list/DISFA_combine_2_3')
    parser.add_argument('--flip_reflect', type=str, default='data1/list/reflect_49.txt')
    parser.add_argument('--res_path_prefix', type=str, default='results/base_jaa1/')
    config = parser.parse_args()

    #net = AUNet(config)
    #net.load_state_dict(torch.load(config.pretrain_prefix + '/au_net.pth'), strict=False)
    ckpt = torch.load(config.pretrain_prefix + '/au_net.pth')
    print(ckpt.keys())