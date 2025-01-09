import numpy as np
from math import cos, sin
import torch
from torch import sin as tsin
from torch import cos as tcos

def Eular_angles_to_rotation_wuhan(angle):
    rot_mat = compute_mat_torch([torch.zeros(1).cuda(), torch.zeros(1).cuda(), angle[2]], 
                                adjust_roll=angle[0], adjust_pitch=angle[1])
    return rot_mat

def compute_mat_torch(angles, adjust_roll=torch.zeros(1).cuda(), adjust_pitch=torch.zeros(1).cuda(), adjust_heading=torch.zeros(1).cuda()):
    roll_ = angles[0]
    pitch_ = -angles[1]
    heading_ = angles[2] - 90.0
    adjust_pitch = -adjust_pitch
    
    R = sub_compute_torch(roll_, pitch_, heading_, adjust_roll, adjust_pitch, adjust_heading)
    R_inv = R.t()
    return R_inv
 
def sub_compute_torch(pitch, roll, heading, t_x, t_y, t_z):
    pitch_rad = torch.deg2rad(pitch)
    roll_rad = torch.deg2rad(roll)
    heading_rad = torch.deg2rad(-heading).unsqueeze(0)
    delta_pitch_rad = torch.deg2rad(t_x).unsqueeze(0)
    delta_roll_rad = torch.deg2rad(t_y).unsqueeze(0)
    delta_heading_rad = torch.deg2rad(t_z)

    tensor_0 = torch.zeros(1).float().cuda()
    tensor_1 = torch.ones(1).float().cuda()
    r1 = torch.stack([
                torch.stack([tensor_1.clone(), tensor_0.clone(), tensor_0.clone()]),
                torch.stack([tensor_0.clone(), tcos(pitch_rad), -tsin(pitch_rad)]),
                torch.stack([tensor_0, tsin(pitch_rad), tcos(pitch_rad)])]).reshape(3,3)
    r2 = torch.stack([
                torch.stack([tcos(roll_rad), tensor_0.clone(), tsin(roll_rad)]),
                torch.stack([tensor_0.clone(), tensor_1.clone(), tensor_0.clone()]),
                torch.stack([-tsin(roll_rad), tensor_0.clone(), tcos(roll_rad)])]).reshape(3,3)
    r3 = torch.stack([
        torch.stack([tcos(heading_rad), -tsin(heading_rad), tensor_0.clone()]),
        torch.stack([tsin(heading_rad), tcos(heading_rad), tensor_0.clone()]),
        torch.stack([tensor_0.clone(), tensor_0.clone(), tensor_1.clone()])
    ]).reshape(3, 3)
    r4 = torch.stack([
        torch.stack([tensor_1.clone(), tensor_0.clone(), tensor_0.clone()]),
        torch.stack([tensor_0.clone(), tcos(delta_pitch_rad), -tsin(delta_pitch_rad)]),
        torch.stack([tensor_0.clone(), tsin(delta_pitch_rad), tcos(delta_pitch_rad)])
    ]).reshape(3, 3)
    r5 = torch.stack([
        torch.stack([tcos(delta_roll_rad), tensor_0.clone(), tsin(delta_roll_rad)]),
        torch.stack([tensor_0.clone(), tensor_1.clone(), tensor_0.clone()]),
        torch.stack([-tsin(delta_roll_rad), tensor_0.clone(), tcos(delta_roll_rad)])
    ]).reshape(3, 3)
    r6 = torch.stack([
        torch.stack([tcos(delta_heading_rad), -tsin(delta_heading_rad), tensor_0.clone()]),
        torch.stack([tsin(delta_heading_rad), tcos(delta_heading_rad), tensor_0.clone()]),
        torch.stack([tensor_0.clone(), tensor_0.clone(), tensor_1.clone()])
    ]).reshape(3, 3)
    R = torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.matmul(r1, r2), r3), r4), r5), r6)
    return R

def build_ray_dir(H, W):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    theta = 360 * (1 - X / W) - 180
    phi = 180 * Y / H
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    z = np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    x = np.cos(theta_rad) * np.sin(phi_rad)
    XYZ = np.concatenate((x[:, :, None], y[:, :, None], z[:, :, None]), axis=-1)
    return XYZ

def sub_compute(pitch, roll, heading, t_x, t_y, t_z):
    r1 = np.array([[1, 0, 0],
                   [0, cos(pitch * np.pi / 180.0), -sin(pitch * np.pi / 180.0)],
                   [0, sin(pitch * np.pi / 180.0), cos(pitch * np.pi / 180.0)]])
    r2 = np.array([[cos(roll * np.pi / 180.0), 0, sin(roll * np.pi / 180.0)],
                   [0, 1, 0],
                   [-sin(roll * np.pi / 180.0), 0, cos(roll * np.pi / 180.0)]])
    r3 = np.array([[cos(-heading * np.pi / 180.0), -sin(-heading * np.pi / 180.0), 0],
                   [sin(-heading * np.pi / 180.0), cos(-heading * np.pi / 180.0), 0],
                   [0, 0, 1]])

    delta_pitch = t_x
    delta_roll = t_y
    delta_heading = t_z###

    r4 = np.array([[1, 0, 0],
                   [0, cos(delta_pitch * np.pi / 180.0), -sin(delta_pitch * np.pi / 180.0)],
                   [0, sin(delta_pitch * np.pi / 180.0), cos(delta_pitch * np.pi / 180.0)]])
    r5 = np.array([[cos(delta_roll * np.pi / 180.0), 0, sin(delta_roll * np.pi / 180.0)],
                   [0, 1, 0],
                   [-sin(delta_roll * np.pi / 180.0), 0, cos(delta_roll * np.pi / 180.0)]])
    r6 = np.array([[cos(delta_heading * np.pi / 180.0), -sin(delta_heading * np.pi / 180.0), 0],
                   [sin(delta_heading * np.pi / 180.0), cos(delta_heading * np.pi / 180.0), 0],
                   [0, 0, 1]])

    R = r1.dot(r2).dot(r3).dot(r4).dot(r5).dot(r6)

    return R

def compute_mat(angles, adjust_roll=0.0, adjust_pitch=0.0, adjust_heading=0.0):
    roll_ = angles[0]
    pitch_ = -angles[1]
    heading_ = angles[2]-90.0
    adjust_pitch = -adjust_pitch
    R = sub_compute(roll_, pitch_, heading_, adjust_roll, adjust_pitch, adjust_heading)
    return np.linalg.inv(R)

def project_3dpc_to_imageuv(pc, proj_param, image_width=2048, get_pc_rect=False):
    coord = proj_param['coord']
    angle = proj_param['angle']
    rot_mat = compute_mat([0.0, 0.0, angle[2]], adjust_roll=angle[0], adjust_pitch=angle[1])
    pts_before_proj = pc - coord
    pts_camera = rot_mat.dot(np.transpose(pts_before_proj))
    pts_camera = np.transpose(pts_camera)
    theta = np.arctan2(pts_camera[:, 1], pts_camera[:, 0]) * 180.0 / np.pi + 180.0
    pts_2d = pts_camera[:, 0:2]
    pts_norm = np.linalg.norm(pts_2d, axis=1)
    phi = np.arctan2(pts_norm, pts_camera[:, 2]) * 180.0 / np.pi

    image_height = image_width / 2
    uv_x = image_width - (theta / 360.0) * image_width
    uv_y = phi / 180.0 * image_height
    uv_x = np.reshape(uv_x.astype(int), [-1, 1])
    uv_y = np.reshape(uv_y.astype(int), [-1, 1])

    if get_pc_rect:
        rect_x = -pts_camera[:,1].reshape((-1,1))   # x->right
        rect_y = -pts_camera[:,2].reshape((-1,1))   # y->down
        rect_z = pts_camera[:,0].reshape((-1,1))    # z->forward
        return np.concatenate([uv_x, uv_y], axis=-1), np.concatenate([rect_x, rect_y, rect_z], axis=-1)
    else:
        return np.concatenate([uv_x, uv_y], axis=-1)

def caculate_one_T(cam_info):
    cam_coord = np.array([float(cam_info[1]), float(cam_info[2]), float(cam_info[3])+0.5])  
    cam_angle = np.array([float(cam_info[4]), float(cam_info[5]), float(cam_info[6])-1.2])  
    
    proj_param = {'coord':cam_coord, 'angle':cam_angle}
    
    coord = proj_param['coord'] # camera xyz
    angle = proj_param['angle'] # camera angle
    
    rot_mat = compute_mat([0.0, 0.0, angle[2]], adjust_roll=angle[0], adjust_pitch=angle[1]) 
    
    rot_mat = rot_mat.T
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot_mat
    transformation_matrix[:3, 3] = coord
    
    return transformation_matrix

def caculate_T(cam_infos):
    transformation_matrix_dic = {}
    for idx in range(len(cam_infos)):
        cam_info = cam_infos[idx]
        cam_id = int(cam_info[0][-4:])
        transformation_matrix_dic[cam_id] = caculate_one_T(cam_info)
    return transformation_matrix_dic


def cam_info_dic(cam_infos):
    out = {}
    for idx in range(len(cam_infos)):
        cam_info = cam_infos[idx]
        cam_id = int(cam_info[0][-4:])
        
        cam_coord = np.array([float(cam_info[1]), float(cam_info[2]), float(cam_info[3])+0.5])
        cam_angle = np.array([float(cam_info[4]), float(cam_info[5]), float(cam_info[6])-1.2])
        
        out[cam_id] = np.concatenate((cam_coord, cam_angle), axis=0)
    return out

def deal_one_line(line):
    return_list = []
    flag = True
    for start_end in line:
        if start_end == "":
            break
        elif start_end == "del":
            flag = False
            continue
        if flag:
            start, end = start_end.split("-")
            return_list += range(int(start), int(end)+1)
        else:
            id = int(start_end)
            return_list.remove(id)
    return return_list

import csv
def sceneId_to_imageId(sceneId=2, filename="data/whu3d/scene.csv"):
    IDs_dic = {}
    with open(filename, "r", encoding="gbk") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            IDs_dic[int(row[0])] = deal_one_line(row[1:])
    return IDs_dic[sceneId]

def merge_nerf_superpoint_sem(nerf_semantic, super_semantic, 
                              ground_label = torch.from_numpy(np.asarray([0, 1, 8])).cuda(),
                              object_label = None):
    
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
    super_label = logits_2_label(super_semantic)
    nerf_label = logits_2_label(nerf_semantic)
    final_tensor = nerf_semantic.clone()

    mask_sp = torch.isin(super_label, ground_label)
    mask_nerf = torch.isin(nerf_label, ground_label)
    mask = mask_sp & mask_nerf

    final_tensor = torch.where(mask_sp, super_label, nerf_label)
    final_tensor = torch.where(mask, nerf_label, final_tensor)
    return final_tensor