from src.models.components.nerf_mlp import get_embedder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mlp_decoder(nn.Module):
    def __init__(self, fr_rgb=15, D=8, W=256, class_num=19, skips=[4], pts_feat_num=32, use_point_color=True, use_point_intensity = True, args=None):
        super(mlp_decoder, self).__init__()
        self.args = args
        self.skips = skips
        self.use_point_color = use_point_color
        self.use_point_intensity = use_point_intensity
        
        counts = np.asarray([use_point_color, use_point_intensity]).sum()
        
        if use_point_color:
            self.pe_rgb, input_ch_rgb = get_embedder(fr_rgb, 0)
            self.rgb_linear = nn.Linear(input_ch_rgb, pts_feat_num)
        
        if use_point_intensity:
            self.pe_i, input_ch = get_embedder(fr_rgb, 0)
            input_ch = input_ch // 3
            self.intensity_linear = nn.Linear(input_ch, pts_feat_num)
        
        first_layer = (counts + 2) * pts_feat_num
            
        self.pe_h, input_ch = get_embedder(fr_rgb, 0)
        input_ch = input_ch // 3
        
        self.height_linear = nn.Linear(input_ch, pts_feat_num)
            
        self.pts_linears = nn.ModuleList(
            [nn.Linear(first_layer, W)] + \
            [nn.Linear(W, W) if i not in self.skips else \
             nn.Linear(W + pts_feat_num, W) for i in range(D-1)])

        self.semantic_output1 = nn.Linear(W, W//2)
        self.semantic_output2 = nn.Linear(W//2, class_num)
    
    def forward(self, height, pts_feats, rgb=None, intensity=None):
        input_height = self.pe_h(height)
        input_height = self.height_linear(input_height)
        
        if self.use_point_color:
            input_rgb = self.pe_rgb(rgb)
            input_rgb = self.rgb_linear(input_rgb)
        
        if self.use_point_intensity:
            input_intensity = self.pe_i(intensity)
            input_intensity = self.intensity_linear(input_intensity)
        
        if self.use_point_color and (self.use_point_intensity == False):
            h = torch.cat((input_rgb, input_height, pts_feats), dim=-1)
        elif (self.use_point_color == False) and self.use_point_intensity:
            h = torch.cat((input_intensity, input_height, pts_feats), dim=-1)
        elif self.use_point_color and self.use_point_intensity:
            h = torch.cat((input_rgb, input_intensity, input_height, pts_feats), dim=-1)
        else:
            h = torch.cat((input_height, pts_feats), dim=-1)
            
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts_feats, h], -1)
        
        semantic = self.semantic_output1(h)
        semantic = self.semantic_output2(F.relu(semantic))

        return semantic