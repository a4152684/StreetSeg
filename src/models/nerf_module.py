from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from src.utils.wuhan_utils import merge_nerf_superpoint_sem
import numpy as np
import os
import laspy
from plyfile import PlyData
from src.utils.utils import validate

city_scapes_colors = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [140, 140, 140]
])
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18, -1]
city_scapes_colors = city_scapes_colors[label_list]

ground_list = np.asarray([0, 1, 7])

city_scapes_colors_whu = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 0, 142],
    [140, 140, 140]
])

city_scapes_colors_kitti360 = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 0, 142],
    [140, 140, 140]
])

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def write_las(data, output_path="output.las", offsets = None):
    out_file = laspy.create(file_version="1.2", point_format=2)
    
    if offsets is not None:
        out_file.offsets = offsets
    
    out_file.x = data[:, 0]
    out_file.y = data[:, 1]
    out_file.z = data[:, 2]
    
    out_file.red = (data[:, 3]).astype(np.uint16)
    out_file.green = (data[:, 4]).astype(np.uint16)
    out_file.blue = (data[:, 5]).astype(np.uint16)
    
    out_file.classification = data[:, 6].astype(np.uint8)

    out_file.write(output_path)

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

class NerfModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        
        self.logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, batch: Any):
        return self.net(batch)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):  
        loss, scalar_stats = self.forward(batch)
        return loss, scalar_stats

    def training_step(self, batch: Any, batch_idx: int):
        loss, scalar_stats = self.model_step(batch)

        # update and log metrics
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("psnr", scalar_stats['psnr_0'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("rgb", scalar_stats['color_mse_0'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("semantic", scalar_stats['semantic_loss_2d_pred'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("depth", scalar_stats['depth_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("cd_p", scalar_stats['cd_p'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("kl", scalar_stats['kl'], on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass
    
    def test_step(self, batch: Any, batch_idx: int):
        if self.net.args.dataset == "wuhan":
            self.test_step_wuhan(batch, batch_idx)
        elif self.net.args.dataset == "kitti360":
            self.test_step_kitti360(batch, batch_idx)

    def test_step_kitti360(self, batch: Any, batch_idx: int):
        try:
            if self.net.args.center_pose == False:
                self.net.args.center_pose = batch['center_pose'][0].float().cpu().numpy()
                self.net.args.dist = batch['dist'][0].float().cpu().numpy()
                print("done")
        except:
            pass
        
        points_dir = self.net.args.points_root
        out_dir = self.net.args.output_root
        filenames = []
        for filename in  os.listdir(points_dir):
            start = int(filename.split("_")[0])
            end = int(filename.split("_")[1].split(".")[0])
            if start >= self.net.args.start and end <= self.net.args.end:
                filenames.append(filename)
        filenames = [os.path.join(points_dir, filename) for filename in filenames]
        
        for filename in filenames:
            points_with_index_filename = "points_with_index.npy"
            points_with_index_filename = os.path.join(self.net.args.visible_root, str(self.net.args.start) +
                                                      "-" + str(self.net.args.end), points_with_index_filename)
            
            index, is_visible, all_points, all_h, rgb, N = self.read_points_with_index(points_with_index_filename, "kitti360")

            semantic_list = []
            all_semantic = []
            chunk = 100000
            for i in range(len(all_points)//chunk + 1):
                points = all_points[i*chunk:(i+1)*chunk]
                semantic = self.net.nerf_0.forward_semantic(points.clone()).squeeze()
                all_semantic.append(semantic)
            all_semantic = torch.cat(all_semantic,0)
            
            vis_points = all_points[is_visible]
            for i in range(len(vis_points)//chunk + 1):
                points = vis_points[i*chunk:(i+1)*chunk]
                semantic = self.net.nerf_0.forward_semantic(points).squeeze()       
                semantic_list.append(semantic)
            
            all_feats = torch.zeros(N, 32).cuda()
            num_superpoints = torch.max(index) + 1
            for i in range(num_superpoints):
                class_indices = torch.where(index == i)[0]
                class_points = all_points[class_indices]
                class_points_mean = torch.mean(class_points, dim=0)
                class_points = class_points - class_points_mean 
                _, class_features = self.net.encoder(class_points.unsqueeze(0))
                all_feats[class_indices] = class_features[0]

            if self.net.args.use_point_color:
                super_predict_semantic = self.net.mlp_decoder(all_h.unsqueeze(1), all_feats, rgb)
            else:
                super_predict_semantic = self.net.mlp_decoder(all_h.unsqueeze(1), all_feats)

            semantic_list.append(super_predict_semantic)
            semantic = torch.cat(semantic_list, 0)
            sem_label = self.logits_2_label(semantic)
            shuffle_index = torch.cat((index[is_visible], index),0)
            new_label = assign_group_labels(shuffle_index, sem_label)[len(index[is_visible]):]
            
            ground_label = torch.from_numpy(np.asarray([0, 1])).cuda()
            all_old_label = self.logits_2_label(all_semantic)
            all_old_label = torch.cat((all_old_label[is_visible], all_old_label[~is_visible]),0)
            final_label = merge_labels(ground_label, all_old_label, new_label)
            final_out_filename = os.path.join(out_dir, filename.split("/")[-1][:-4]+".las")
            points = all_points.clone()
            self.write_to_file(final_out_filename, points, final_label, "kitti360")
        
            name_list = ['road', 
             'sidewalk',
             'building',
             'wall',
             'fence',
             'pole', 
             'traffic sign', 
             'vegetation', 
             'terrian',
             'car']
            gt_filename = os.path.join(self.net.args.points_root, str(self.net.args.scene) + ".las")
            gt_label = np.asarray(laspy.read(gt_filename).classification)
            gt_label = torch.from_numpy(gt_label)
            mask = (gt_label != self.net.args.class_num-1)
            validate(final_label[mask], gt_label[mask], name_list, out_classes=self.net.args.class_num-1)
            
        exit()

    
    def write_to_file(self, out_filename, points, label, dataset="whu"):
        points = points.squeeze().cpu().numpy()
        points = self.net.args.dist * points
        points = points + np.asarray(self.net.args.center_pose)
        points = torch.from_numpy(points).cuda()
        
        if dataset == "whu":
            colour_map_np = torch.from_numpy(np.asarray(city_scapes_colors_whu)).cuda()
        elif dataset == "kitti360":
            colour_map_np = torch.from_numpy(np.asarray(city_scapes_colors_kitti360)).cuda()
            
        vis_sem_label = colour_map_np[label].int()
        new_output = torch.cat([points, vis_sem_label, label.unsqueeze(1)], dim=1)
        write_las(new_output.cpu().numpy(), out_filename)
        print(out_filename)
    
    def read_points_with_index(self, points_with_index_filename, dataset="whu"):
        if dataset == "whu":
            all_points = np.load(points_with_index_filename)
            N = all_points.shape[0]
            
            index = all_points[:, 0].astype(int)
            is_visible = all_points[:, 4].astype(bool)
            all_rgb = all_points[:, 6:9]
            all_h = all_points[:, 5]
            all_intensity = all_points[:, 9]
            all_points = all_points[:, 1:4]

            # 构造高度特征
            all_h = all_h / self.net.args.height_scale
            
            all_points = all_points - np.asarray(self.net.args.center_pose)
            all_points /= self.net.args.dist
            
            index = torch.from_numpy(index).cuda()
            is_visible = torch.from_numpy(is_visible).cuda()
            all_points = torch.from_numpy(all_points).float().cuda()
            all_h = torch.from_numpy(all_h).float().cuda().unsqueeze(1)
            all_rgb = torch.from_numpy(all_rgb).float().cuda()
            all_intensity = torch.from_numpy(all_intensity).float().cuda().unsqueeze(1)
            
            return index, is_visible, all_points, all_h, all_rgb, all_intensity, N
        
        elif dataset == "kitti360":
            all_points = np.load(points_with_index_filename)
            N = all_points.shape[0]
            
            index = all_points[:, 0].astype(int)
            is_visible = all_points[:, 4].astype(bool)
            rgb = all_points[:, 5:8]
            all_h = all_points[:, -1]
            all_points = all_points[:, 1:4]
            
            # 构造高度特征
            all_h = all_h / self.net.args.height_scale
            
            # 颜色归一化，位置中心归一化
            rgb /= 255
            
            all_points = all_points - np.asarray(self.net.args.center_pose)
            all_points /= self.net.args.dist
            
            index = torch.from_numpy(index).cuda()
            is_visible = torch.from_numpy(is_visible).cuda()
            all_points = torch.from_numpy(all_points).float().cuda()
            all_h = torch.from_numpy(all_h).float().cuda()
            rgb = torch.from_numpy(rgb).float().cuda()
            return index, is_visible, all_points, all_h, rgb, N
            
    def test_step_wuhan(self, batch: Any, batch_idx: int):
        try:
            if self.net.args.center_pose == False:
                self.net.args.center_pose = batch['center_pose'][0].float().cpu().numpy()
                self.net.args.dist = batch['dist'][0].float().cpu().numpy()
                print("done")
        except:
            pass

        out_dir = self.net.args.output_root
        
        filenames = [str(self.net.args.scene)]
        
        for filename in filenames:
            points_with_index_filename = "points_with_index.npy"
            points_with_index_filename = os.path.join(self.net.args.superpoint_root, 
                                                      str(self.net.args.scene) + "_supervoxels", 
                                                      points_with_index_filename)
            index, _, all_points, all_h, all_rgb, all_intensity, N = self.read_points_with_index(
                points_with_index_filename, "whu")

            all_semantic = []
            chunk = 100000
            for i in range(len(all_points)//chunk + 1):
                points = all_points[i*chunk:(i+1)*chunk]
                semantic = self.net.nerf_0.forward_semantic(points).squeeze()
                all_semantic.append(semantic)
            nerf_semantic = torch.cat(all_semantic, 0)
            
            superpoint_semantic = []
            all_feats = torch.zeros(N, 128).cuda()
            num_superpoints = torch.max(index) + 1
            for i in range(num_superpoints):
                class_indices = torch.where(index == i)[0]
                class_points = all_points[class_indices]
                class_points_mean = torch.mean(class_points, dim=0)
                class_points = class_points - class_points_mean 
                class_points *= torch.from_numpy(self.net.args.dist).cuda()
                _, class_features = self.net.encoder(class_points.unsqueeze(0))
                all_feats[class_indices] = class_features[0]
            
            for i in range(len(all_h)//chunk + 1):
                if self.net.args.use_point_color and self.net.args.use_point_intensity:
                    super_predict_semantic = self.net.mlp_decoder(all_h[i*chunk:(i+1)*chunk], 
                                                                  all_feats[i*chunk:(i+1)*chunk], 
                                                                  rgb=all_rgb[i*chunk:(i+1)*chunk], 
                                                                  intensity=all_intensity[i*chunk:(i+1)*chunk])
                elif self.net.args.use_point_color and (self.net.args.use_point_intensity == False):
                    super_predict_semantic = self.net.mlp_decoder(all_h[i*chunk:(i+1)*chunk], 
                                                                  all_feats[i*chunk:(i+1)*chunk], 
                                                                  rgb=all_rgb[i*chunk:(i+1)*chunk])
                elif (self.net.args.use_point_color==False) and self.net.args.use_point_intensity:
                    super_predict_semantic = self.net.mlp_decoder(all_h[i*chunk:(i+1)*chunk], 
                                                                  all_feats[i*chunk:(i+1)*chunk], 
                                                                  intensity=all_intensity[i*chunk:(i+1)*chunk])
                else:
                    super_predict_semantic = self.net.mlp_decoder(all_h[i*chunk:(i+1)*chunk], 
                                                                  all_feats[i*chunk:(i+1)*chunk])
                    
                all_semantic.append(super_predict_semantic)
                superpoint_semantic.append(super_predict_semantic)
                
            superpoint_semantic= torch.cat(superpoint_semantic, 0)
            sem_label = merge_nerf_superpoint_sem(nerf_semantic, superpoint_semantic, torch.from_numpy(np.asarray([0, 1])).cuda())
            sem_label = assign_group_labels(index, sem_label)

            ground_label = torch.from_numpy(ground_list).cuda()
            nerf_label = self.logits_2_label(nerf_semantic)
            final_label = merge_labels(ground_label, nerf_label, sem_label)
            final_out_filename = os.path.join(out_dir, filename+".las")
            points = all_points.clone()
            self.write_to_file(final_out_filename, points, final_label, "whu")
        
        name_list = ['road', 
             'sidewalk',
             'fence',
             'pole', 
             'traffic light', 
             'traffic sign', 
             'vegetation', 
             'terrian',
             'car']
        gt_filename = os.path.join(self.net.args.points_root, str(self.net.args.scene) + ".las")
        gt_label = np.asarray(laspy.read(gt_filename).classification)
        gt_label = torch.from_numpy(gt_label)
        mask = (gt_label != self.net.args.class_num-1)
        validate(final_label[mask], gt_label[mask], name_list, out_classes=self.net.args.class_num-1)
        exit()
           
    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        # 改，加个正则化
        # optimizer = self.hparams.optimizer(params=self.parameters(), weight_decay=1e-2) 
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def semantic2label(self, outputs, semantic, points, dataset="kitti360"):
        logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        sem_label = logits_2_label(semantic)
        origin_sem_label = sem_label.clone().unsqueeze(1) # N 1
        
        if dataset == "whu3d":
            colour_map_np = torch.from_numpy(np.asarray(city_scapes_colors_whu)).cuda()
        elif dataset == "kitti360":
            colour_map_np = torch.from_numpy(np.asarray(city_scapes_colors_kitti360)).cuda()
        
        vis_sem_label = colour_map_np[sem_label].int()
        points = points.squeeze().cpu().numpy()
        points = self.net.args.dist * points
        points = points + np.asarray(self.net.args.center_pose)
        points = torch.from_numpy(points).cuda()
        outputs.append(torch.cat([points, vis_sem_label, origin_sem_label], dim=1))
        
        return outputs

def assign_group_labels(index, label):
    num_groups = torch.max(index) + 1
    num_labels = torch.max(label) + 1
    label_counts = torch.zeros(num_groups, num_labels).long().cuda()

    for i in range(num_groups):
        group_indices = torch.where(index == i)[0]
        group_labels = label[group_indices]
        unique_labels, counts = torch.unique(group_labels, return_counts=True)
        label_counts[i, unique_labels] = counts

    max_label_indices = torch.argmax(label_counts, dim=1)
    new_label = max_label_indices[index]
    return new_label

import torch
import torch.nn.functional as F
def merge_labels(ground_label, all_old_label, all_new_label):
    final_tensor = all_new_label.clone()
    mask = torch.isin(final_tensor, ground_label)
    a_model_labels = all_old_label[mask]
    matching_labels = torch.isin(a_model_labels, ground_label)
    final_tensor[mask] = torch.where(matching_labels, a_model_labels, final_tensor[mask])
    return final_tensor
