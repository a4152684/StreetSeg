import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from src.utils.nerf_utils import raw2outputs_semantic_joint
from copy import deepcopy
from plyfile import PlyData
import os
from src.utils.pointnet2_utils import pointnet_encoder, pointnet_decoder
from src.utils.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamLoss = chamfer_3DDist()

from src.utils.wuhan_utils import Eular_angles_to_rotation_wuhan

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    r = plydata['vertex']['red']
    g = plydata['vertex']['green']
    b = plydata['vertex']['blue']
    is_visible = plydata['vertex']['visible'].astype(bool)
    points = np.stack((x, y, z), axis=-1)
    rgb = np.stack((r, g, b), axis=-1)
    return points, rgb, is_visible

class Network(nn.Module):
    def __init__(self, nerf_0, mlp_decoder, args):
        super(Network, self).__init__()
        self.nerf_0 = nerf_0
        self.mlp_decoder = mlp_decoder
        self.args = args
        self.color_crit = nn.MSELoss(reduction='mean')
        self.depth_crit = nn.HuberLoss(reduction='mean')
        self.nll = nn.NLLLoss()
        if self.args.dataset == "wuhan":
            self.eular_to_rot = Eular_angles_to_rotation_wuhan

        superpoint_filename = "superpoint_train.npy"
        if self.args.dataset == "kitti360":
            superpoint_filename = os.path.join(self.args.superpoint_root, str(self.args.start) + "-" + str(self.args.end), superpoint_filename)
            self.super_points = np.load(superpoint_filename) # x y z visible height r g b
        elif self.args.dataset == "wuhan":
            superpoint_filename = os.path.join(self.args.superpoint_root, str(self.args.scene) + "_supervoxels", superpoint_filename)
            self.super_points = np.load(superpoint_filename) # x y z visible height r g b intensity

        self.super_is_visible = self.super_points[:, :, 3].astype(bool)
        self.h = self.super_points[:, :, 4]
        self.h = self.h / self.args.height_scale
        if self.args.use_point_color:
            self.super_rgb = self.super_points[:, :, 5:8]
            if self.args.dataset == "kitti360":
                self.super_rgb /= 255.
            self.super_rgb = torch.from_numpy(self.super_rgb).float().cuda()
        if self.args.use_point_intensity:
            self.super_intensity = self.super_points[:, :, 8]
            self.super_intensity = torch.from_numpy(self.super_intensity).float().cuda()[:, :, None]

        self.super_points = self.super_points[:, :, :3]
        self.super_points = torch.from_numpy(self.super_points).float().cuda() # B M 3
        self.h= torch.from_numpy(self.h).float().cuda()[:, :, None] # B M 1
        self.super_is_visible = torch.from_numpy(self.super_is_visible).cuda()

        self.encoder = pointnet_encoder(args.pts_feats_num)
        self.decoder = pointnet_decoder(pts_feats_num=args.pts_feats_num, num_coarse=args.pts_num_per_sp)

        self.super_is_visible_flatten = self.super_is_visible.reshape(-1)

        if self.args.dataset == "wuhan":
            self.pose_offset = nn.Parameter(torch.randn(6)/100, requires_grad=True)
        
    def render_rays(self, xyz, ray_dir, z_vals):
        
        raw = self.nerf_0(xyz, ray_dir)
        outputs = {}
        ret_0 = raw2outputs_semantic_joint(raw, z_vals)
        for key in ret_0:
            outputs[key + '_0'] = ret_0[key]
        return outputs 

    def batchify_rays(self, xyz, ray_dir, z_vals):
        all_ret = {}
        chunk = self.args.chunk_size
        for i in range(0, xyz.shape[1], chunk):
            ret = self.render_rays(xyz[:,i:i+chunk], ray_dir[:,i:i+chunk], z_vals[:,i:i+chunk])
            torch.cuda.empty_cache()
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret
    
    def forward_semantic(self, point_cloud, rgb, pts_feat):
        sem_mlp = self.mlp_decoder(point_cloud, rgb, pts_feat)
        sem_nerf = self.nerf_0.forward_semantic(point_cloud, pts_feat)
        return sem_mlp, sem_nerf
        
    def forward(self, batch):
        try:
            if self.args.center_pose == False:
                self.args.center_pose = batch['center_pose'][0].float()
                self.args.dist = batch['dist'][0].float()
                self.super_points = self.super_points - self.args.center_pose
                self.super_points /= self.args.dist
                
                _, M, _ = self.super_points.shape
                self.super_points_mean = torch.mean(self.super_points, dim=1)
                self.super_points_no_center = self.super_points - self.super_points_mean[:, None, :].repeat(1, M, 1)
                
                if self.args.is_sp_log:
                    B = self.super_points.shape[0]
                    random_indices = torch.randperm(B)[:self.args.random_superpoint_nums].cuda()
                    self.super_points_train = self.super_points[random_indices]
                    self.super_points_no_center_train = self.super_points_no_center[random_indices]
                    self.h_train = self.h[random_indices]
                    self.super_is_visible_train = self.super_is_visible[random_indices]
                    self.super_is_visible_flatten_train = self.super_is_visible_train.reshape(-1)
                    if self.args.use_point_color:
                        self.super_rgb_train = self.super_rgb[random_indices]
                    if self.args.use_point_intensity:
                        self.super_intensity_train = self.super_intensity[random_indices]
                
                print("initialization done")
        except:
            pass

        if self.args.dataset == "wuhan":
            ray_dir = batch['ray_dir'] # B N 3
            pose = batch['pose']
            pose = pose + self.pose_offset # B 6
            rot_mat = []
            for i in range(len(pose)):
                temp = self.eular_to_rot(pose[i, 3:])
                rot_mat.append(temp.unsqueeze(0))
            rot_mat = torch.cat(rot_mat, 0) # B 3 3
        
            ray_dir = torch.bmm(ray_dir, rot_mat) # B N 3
            rays_o = pose[:, :3].unsqueeze(1).repeat(1, ray_dir.shape[1], 1) # B N 3
            ray = torch.cat((rays_o, ray_dir), dim=-1) # B N 6
            
        elif self.args.dataset == "kitti360":
            ray = batch['ray']
            
        depth = batch['depth']
        xyz, ray_dir, z_vals = self.build_ray_points(ray, depth)
        ret = self.batchify_rays(xyz, ray_dir, z_vals)
        
        B = self.super_points.shape[0]
        random_indices = torch.randperm(B)[:self.args.random_superpoint_nums].cuda()
        self.super_points_train = self.super_points[random_indices]
        self.super_points_no_center_train = self.super_points_no_center[random_indices]
        self.h_train = self.h[random_indices]
        self.super_is_visible_train = self.super_is_visible[random_indices]
        self.super_is_visible_flatten_train = self.super_is_visible_train.reshape(-1)
        if self.args.use_point_color:
            self.super_rgb_train = self.super_rgb[random_indices]
        if self.args.use_point_intensity:
            self.super_intensity_train = self.super_intensity[random_indices]
            
        super_semantic = self.nerf_0.forward_semantic(self.super_points_train)
        super_semantic = deepcopy(super_semantic.detach()).requires_grad_()
        super_global_feats, super_pts_feats = self.encoder(self.super_points_no_center_train)
        reconstruct_super_points = self.decoder(super_global_feats)

        if self.args.use_point_color and self.args.use_point_intensity:
            super_predict_semantic = self.mlp_decoder(self.h_train, super_pts_feats,rgb=self.super_rgb_train, intensity=self.super_intensity_train)
        elif self.args.use_point_color and (self.args.use_point_intensity == False):
            super_predict_semantic = self.mlp_decoder(self.h_train, super_pts_feats, rgb=self.super_rgb_train)
        elif (self.args.use_point_color==False) and self.args.use_point_intensity:
            super_predict_semantic = self.mlp_decoder(self.h_train, super_pts_feats, intensity=self.super_intensity_train)
        else:
            super_predict_semantic = self.mlp_decoder(self.h_train, super_pts_feats)
        
        loss, scalar_stats = self.compute_loss(batch, ret, super_semantic, self.super_points_no_center_train, 
                                               reconstruct_super_points, super_predict_semantic, self.super_is_visible_flatten_train)
        
        
        return loss, scalar_stats
    
    def compute_loss(self, batch, output, super_semantic, super_points_no_center, reconstruct_super_points, super_predict_semantic, super_is_visible_flatten):
        scalar_stats = {}
        loss = 0
    
        # rgb loss
        if 'rgb_0' in output.keys():
            color_loss = self.args.train_params.weight_color * self.color_crit(batch['rays_rgb'], output['rgb_0'])
            scalar_stats.update({'color_mse_0': color_loss})
            loss += color_loss
            psnr = -10. * torch.log(color_loss.detach()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({'psnr_0': psnr})
        
        # depth loss
        if ('depth_0' in output.keys()) and ('depth' in batch):
            pred_depth = output['depth_0'].reshape(-1)
            gt_depth = batch['depth'].reshape(-1)
            gt_depth /= self.args.dist
            mask_depth = ~torch.isinf(gt_depth)
            depth_loss = self.depth_crit(gt_depth[mask_depth], pred_depth[mask_depth])
            scalar_stats.update({'depth_loss': depth_loss})
            loss += self.args.train_params.lambda_depth * depth_loss
            
        # semantic_loss
        if 'semantic_map_0' in output.keys():
            confidence = batch['confidence'] # B N
            semantic_loss = 0.
            decay = 1.
            pseudo_label = batch['pseudo_label']
            _, _, channel = output['semantic_map_0'].shape

            if self.args.use_confidence:
                pred = torch.log(output['semantic_map_0'].reshape(-1, channel)+1e-5)
                gt = pseudo_label.reshape(-1).long()
                confidence = confidence.reshape(-1)
                pred = pred * confidence.unsqueeze(1)
                semantic_loss_2d_pred = self.nll(pred, gt)
            else:
                semantic_loss_2d_pred = self.nll(torch.log(output['semantic_map_0'].reshape(-1, channel)+1e-5), pseudo_label.reshape(-1).long())                
            
            semantic_loss_2d_pred = decay * self.args.train_params.lambda_semantic_2d  * semantic_loss_2d_pred
            semantic_loss += semantic_loss_2d_pred
            
            loss += self.args.train_params.semantic_weight * semantic_loss

            scalar_stats.update({'semantic_loss_2d_pred': semantic_loss_2d_pred})
        
        dist1, dist2, _, _ = chamLoss(super_points_no_center, reconstruct_super_points)
        cd_p_loss = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
        cd_p_loss = torch.mean(cd_p_loss)
        loss += cd_p_loss
        scalar_stats.update({'cd_p': cd_p_loss})
        
        super_predict_semantic = super_predict_semantic.reshape(-1, self.args.class_num)
        super_semantic = super_semantic.reshape(-1, self.args.class_num)
        super_predict_semantic = super_predict_semantic[super_is_visible_flatten]
        super_semantic = super_semantic[super_is_visible_flatten]
        
        kl_loss = F.kl_div(F.log_softmax(super_predict_semantic, dim=-1), 
                           F.softmax(super_semantic, dim=-1), 
                           reduction='mean')
        loss += kl_loss
        scalar_stats.update({'kl': kl_loss})

        scalar_stats.update({'loss': loss})
        
        return loss.mean(), scalar_stats

    def build_ray_points(self, rays, depth):
        rays_o, ray_dir = rays[:, :, :3], rays[:, :, 3:6] # B N 3
        scale_factor = torch.norm(ray_dir, p=2, dim=2)#  B N
        ray_dir = ray_dir / scale_factor[:, :, None] # 
        near, far, samples_all, sample_radius = self.args.near, self.args.far, self.args.samples_all, self.args.sample_radius
        
        xyz, z_vals = self.sample_along_rays(rays_o, ray_dir, depth, near, far, sample_radius, samples_all)
        xyz /= self.args.dist
        z_vals /= self.args.dist
        # ray_dir从 [B N 3] -> [B N samples 3]
        ray_dir = ray_dir.unsqueeze(-2).repeat(1,1,samples_all,1)
        return xyz, ray_dir, z_vals
    
    def sample_along_rays(self, rays_o, rays_d, depth, near=0., far=1., sample_radius=2., samples=64):
        # rays_o: B x N x 3, rays_d: B x N x 3, depth: B x N
        # near, far: scalar, samples: scalar
        # create a mask to identify rays with infinite depth
        inf_mask = torch.isinf(depth)
        finite_mask = ~inf_mask
        # create a tensor to store the sampled points
        sampled_points = torch.zeros(rays_o.shape[:-1] + (samples, 3), device=rays_o.device, requires_grad=True)
        # B N samples 3
        sampled_z_vals = torch.zeros(rays_o.shape[:-1] + (samples, 1), device=rays_o.device, requires_grad=True).squeeze(-1)
        # B N samples
        # sample points for rays with finite depth
        finite_depths = depth[finite_mask].unsqueeze(-1) # N_finite 1
        finite_rays_o = rays_o[finite_mask] # N_finite 3
        finite_rays_d = rays_d[finite_mask]

        if finite_depths.numel() > 0:
            z_steps = torch.linspace(0, 1, samples, device=rays_o.device, requires_grad=True)
            z_vals = (finite_depths - sample_radius) * (1-z_steps) + (finite_depths + sample_radius) * z_steps
            # N_finite samples
            # x + td
            finite_points = finite_rays_o.unsqueeze(-2) + finite_rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
            #N samples 3        N 1 3                           N 1 3                      N sapmles 1   
            sampled_points_updated = sampled_points.clone()
            sampled_z_vals_updated = sampled_z_vals.clone()
            sampled_points_updated[finite_mask] = finite_points
            sampled_z_vals_updated[finite_mask] = z_vals
            
        # sample points for rays with infinite depth
        if inf_mask.any():
            t_vals = torch.linspace(near, far, samples, device=rays_o.device, requires_grad=True)
            inf_rays_o = rays_o[inf_mask]
            inf_rays_d = rays_d[inf_mask]
            inf_points = inf_rays_o.unsqueeze(-2) + inf_rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)
            sampled_points_updated[inf_mask] = inf_points
            sampled_z_vals_updated[inf_mask] = t_vals

        return sampled_points_updated, sampled_z_vals_updated
    
