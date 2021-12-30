import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from models.jaav2_multi.transformer import TransformerEncoder


class T_net_multi(nn.Module):
    def __init__(self, configs, au_num=8):
        """
        Construct a MulT model.
        """
        super(T_net_multi, self).__init__()
        # setting
        self.ori_dim = 512
        self.proj_dim = configs.proj_dim
        self.emb_dim = self.proj_dim          # emb dim of transformer
        self.comb_emb_dim = self.emb_dim * 5  # cat all
        self.num_heads = configs.num_heads
        self.num_layers = configs.num_layers
        self.attn_mask = configs.attn_mask
        # dropout
        self.attn_dropout = configs.attn_dropout
        self.relu_dropout = configs.relu_dropout
        self.embed_dropout = configs.embed_dropout
        self.res_dropout = configs.res_dropout
        self.out_dropout = configs.out_dropout

        # 1. Temporal convolutional layers , may not useful here
        self.proj_1 = nn.Linear(self.ori_dim, self.proj_dim, bias=False)
        self.proj_2 = nn.Linear(self.ori_dim, self.proj_dim, bias=False)
        self.proj_3 = nn.Linear(self.ori_dim, self.proj_dim, bias=False)
        self.proj_4 = nn.Linear(self.ori_dim, self.proj_dim, bias=False)
        self.proj_5 = nn.Linear(self.ori_dim, self.proj_dim, bias=False)
        # 2. Crossmodal Attentions
        self.trans_1_with_3 = self.get_network(self.emb_dim)
        self.trans_2_with_3 = self.get_network(self.emb_dim)
        self.trans_4_with_3 = self.get_network(self.emb_dim)
        self.trans_5_with_3 = self.get_network(self.emb_dim)
        # 3. Self Attention
        self.trans_3_with_3 = self.get_network(self.emb_dim)
        # 4. Merge
        self.trans_3_mem = self.get_network(self.comb_emb_dim)
        # 5. Residual
        self.proj1 = nn.Linear(self.comb_emb_dim, self.comb_emb_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.out_dropout)
        self.proj2 = nn.Linear(self.comb_emb_dim, self.comb_emb_dim)
        # 6. Output
        self.out_layer = nn.Linear(self.comb_emb_dim, au_num * 2)

    def get_network(self, emb_dim):
        return TransformerEncoder(embed_dim=emb_dim,
                                  num_heads=self.num_heads,
                                  layers=self.num_layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, concat_au_feat_list):
        # project: [B, 512] -> [1, B, proj_dim]
        if self.ori_dim != self.proj_dim:
            feat1 = self.proj_1(concat_au_feat_list[0]).unsqueeze(0)
            feat2 = self.proj_1(concat_au_feat_list[1]).unsqueeze(0)
            feat3 = self.proj_1(concat_au_feat_list[2]).unsqueeze(0)
            feat4 = self.proj_1(concat_au_feat_list[3]).unsqueeze(0)
            feat5 = self.proj_1(concat_au_feat_list[4]).unsqueeze(0)
        else:
            feat1 = concat_au_feat_list[0].unsqueeze(0)
            feat2 = concat_au_feat_list[1].unsqueeze(0)
            feat3 = concat_au_feat_list[2].unsqueeze(0)
            feat4 = concat_au_feat_list[3].unsqueeze(0)
            feat5 = concat_au_feat_list[4].unsqueeze(0)

        # cross attention
        feat_1_3 = self.trans_1_with_3(feat3, feat1, feat1)
        feat_2_3 = self.trans_1_with_3(feat3, feat2, feat2)
        feat_4_3 = self.trans_1_with_3(feat3, feat4, feat4)
        feat_5_3 = self.trans_1_with_3(feat3, feat5, feat5)
        #print('feat_1_3:', feat_1_3.shape)
        # self attention
        feat_3_3 = self.trans_3_with_3(feat3, feat3, feat3)
        #print('feat_3_3:', feat_3_3.shape)
        # merge
        feat_merge = torch.cat([feat_1_3, feat_2_3, feat_3_3, feat_4_3, feat_5_3], dim=2)
        feat_merge = self.trans_3_mem(feat_merge).squeeze(0)
        #print('feat_merge:', feat_merge.shape)

        # residual
        feat_merge_proj = self.proj1(self.dropout1(self.relu1(self.proj1(feat_merge))))
        feat_out = feat_merge + feat_merge_proj
        #print('feat_merge_res:', feat_merge.shape)

        # output
        au_output = self.out_layer(feat_out)
        au_output = au_output.view(au_output.size(0), 2, int(au_output.size(1) / 2))
        au_output = F.log_softmax(au_output, dim=1)
        #print('output:', au_output.shape)
        return au_output


if __name__ == "__main__":
    # transformer encoder
    encoder = TransformerEncoder(
        embed_dim=30,
        num_heads=5,
        layers=5,
        attn_dropout=0.5,
        relu_dropout=0.5,
        res_dropout=0.5,
        embed_dropout=0.5,
        attn_mask=False
    )

    inp = torch.randn(1, 8, 30)
    oup = encoder(inp, inp, inp)

    print(oup.shape)

