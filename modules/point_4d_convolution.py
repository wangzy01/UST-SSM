import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from typing import List
from knn_cuda import KNN
from timm.models.layers import DropPath
from bimamba_ssm.modules.mamba_simple import Mamba
from ipdb import set_trace as st
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            new_feature = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                neighbor_xyz = xyzs[i]

                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                displacement = self.conv_d(displacement)

                if self.in_planes != 0:
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)


                new_feature.append(feature)
            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features




class P4DTransConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 original_planes: int = 0,
                 bias: bool = False):
        """
        Args:
            in_planes: C'. when point features are not available, in_planes is 0.
            out_planes: C"
            original_planes: skip connection from original points. when original point features are not available, original_in_planes is 0.
            bias: whether to use bias
            batch_norm: whether to use batch norm
            activation:
        """
        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm

        conv = []
        for i in range(len(mlp_planes)):
            if i == 0:
                conv.append(nn.Conv1d(in_channels=in_planes+original_planes, out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            else:
                conv.append(nn.Conv1d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                conv.append(nn.BatchNorm1d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, xyzs: torch.Tensor, original_xyzs: torch.Tensor, features: torch.Tensor, original_features: torch.Tensor = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        xyzs : torch.Tensor
            (B, T, N', 3) tensor of the xyz positions of the convolved features
        original_xyzs : torch.Tensor
            (B, T, N, 3) tensor of the xyz positions of the original points
        features : torch.Tensor
            (B, T, C', N') tensor of the features to be propigated to
        original_features : torch.Tensor
            (B, T, C, N) tensor of original point features for skip connection

        Returns
        -------
        new_features : torch.Tensor
            (B, T, C", N) tensor of the features of the unknown features
        """

        T = xyzs.size(1)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
        features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        new_xyzs = original_xyzs

        original_xyzs = torch.split(tensor=original_xyzs, split_size_or_sections=1, dim=1)
        original_xyzs = [torch.squeeze(input=original_xyz, dim=1).contiguous() for original_xyz in original_xyzs]

        if original_features is not None:
            original_features = torch.split(tensor=original_features, split_size_or_sections=1, dim=1)
            original_features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in original_features]

        new_features = []

        for t in range(T):
            dist, idx = pointnet2_utils.three_nn(original_xyzs[t], xyzs[t])

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feat = pointnet2_utils.three_interpolate(features[t], idx, weight)

            if original_features is not None:
                new_feature = torch.cat([interpolated_feat, original_features[t]], dim=1)
            new_feature = self.conv(new_feature)
            new_features.append(new_feature)

        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features


class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        batch_size, num_points, _ = xyz.shape 
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz) 
        assert idx.size(1) == num_points # N center
        assert idx.size(2) == self.group_size # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous() 
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :] 
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous() 
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, neighborhood_feat


class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)
        mean_x = lc_x.unsqueeze(dim=-2) 
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz) 

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5) 

        B, G, K, C = knn_x.shape

        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat 
        
        knn_x_w = knn_x.permute(0, 3, 1, 2) 

        return knn_x_w


class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation
        e_x = torch.exp(knn_x_w) 
        up = (knn_x_w * e_x).mean(-1) 
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        return lc_x


class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute
    
    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)

class STSAL(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128, act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm,):
        super().__init__()
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()
        
    
    def forward(self, center, feat):

        B, G, C = feat.shape
        lc_x_w = self.lga(center, feat)         
        lc_x_w = self.kpool(lc_x_w)

        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1)) 
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1)) 
        
        lc_x = self.act(lc_x)
        
        return lc_x


class AggregationSSM(nn.Module):
    def __init__(self, 
                dim, 
                mlp_ratio=4., 
                drop=0., 
                drop_path=0., 
                act_layer=nn.SiLU, 
                norm_layer=nn.LayerNorm,
                k_group_size=8, 
                alpha=100, 
                beta=1000,
                num_group=128,
                num_heads=6,
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        self.num_group = num_group
        self.k_group_size = k_group_size
        
        self.num_heads = num_heads
        
        self.stsal = STSAL(lga_out_dim=dim*2, 
                    k_group_size=self.k_group_size, 
                    alpha=alpha, 
                    beta=beta, 
                    mlp_in_dim=dim*2, 
                    mlp_out_dim=dim, 
                    num_group=self.num_group,
                    act_layer=act_layer,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    )

        self.mixer = Mamba(dim)

        

    def shuffle_x(self, x, shuffle_idx):
        pos = x[:, None, 0, :]
        feat = x[:, 1:, :]
        shuffle_feat = feat[:, shuffle_idx, :]
        x = torch.cat([pos, shuffle_feat], dim=1)
        return x

    def mamba_shuffle(self, x):
        G = x.shape[1] - 1 
        shuffle_idx = torch.randperm(G)
        x = self.shuffle_x(x, shuffle_idx) 

        x = self.mixer(self.norm2(x))

        x = self.shuffle_x(x, shuffle_idx) 
        return x

    def forward(self, center, x):
        x = x + self.drop_path(self.stsal(center, self.norm1(x))) 
        x = x + self.drop_path(self.mixer(self.norm2(x)))
        return x


