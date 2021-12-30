import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class T_net_lstm(nn.Module):
    def __init__(self, feat_dim=512, num_layers=5, unit_dim=8, au_num=8, bi=False):
        super(T_net_lstm, self).__init__()
        if bi:
            num_layers *= 2

        self.lstm = nn.LSTM(input_size=feat_dim,
                            hidden_size=unit_dim * 64,
                            num_layers=num_layers,
                            bidirectional=bi)

        self.fc2 = nn.Linear(unit_dim * 64, au_num * 2)


    def forward(self, concat_au_feat_list):
        cat_au_feats = torch.cat(concat_au_feat_list, dim=0)
        _, (hn, cn) = self.lstm(cat_au_feats)
        au_output = self.fc2(hn[-1])
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1) / 2))
        au_output = F.log_softmax(au_output, dim=1)

        return au_output

