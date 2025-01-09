from torch.utils.data import Dataset
import numpy as np
import os 
import cv2
from src.utils.wuhan_utils import caculate_T, sceneId_to_imageId, cam_info_dic, build_ray_dir
import laspy
from random import shuffle

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

class WuHanDataset(Dataset):

    def __init__(self, args, split="train"):
        """ Load data from given dataset directory. """
        self.split = split
        self.args = args
        
        self.IDs = sceneId_to_imageId(self.args.scene)
        
        if self.split == "val" or self.split == "test":
            self.IDs = self.IDs[:1]
        
        self.car_mask = np.load(self.args.mask_root) # (4096, 8192)
        self.car_mask = self.car_mask.reshape(-1)

        self.label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18]

        self.training_cityscape_to_new_whu3d = {
                                            0: 0, # road
                                            1: 1, # sidewalk
                                            2: 9, # building
                                            3: 2, # wall
                                            4: 2, # fence
                                            5: 3, # pole
                                            6: 4, # traffic light
                                            7: 5, # traffic sign
                                            8: 6, # vegetation
                                            9: 7, # terrain
                                            10: 9, # person
                                            11: 8,  # car
                                            12: 8,  # truck
                                            13: 8,  # bus
                                            14: 8,  # train
                                            15: 9, # others
                                        }
        
        cam_infos = np.loadtxt(args.caminfo_root, usecols=(0,3,4,5,8,9,10), delimiter=' ', dtype='str') 
        imageId = cam_infos[:,0].astype(np.int32)
        self.cam_infos = cam_info_dic(cam_infos)

        self.c2w_dic = caculate_T(cam_infos)
        self.translation = args.center_pose

        self.images_list = {}
        for i, ID in enumerate(imageId):
            filename = str(int(ID)).zfill(17)
            name = filename + ".jpg"
            name = os.path.join(args.images_root, name)
            self.images_list[int(ID)] = name

        self.sem_list = {}
        for i, ID in enumerate(imageId):
            filename = str(int(ID)).zfill(17)
            name = filename + ".png"
            name = os.path.join(args.sem_root, self.args.pretrained_2d_model, name)
            self.sem_list[int(ID)] = name

        self.con_list = {}
        for i, ID in enumerate(imageId):
            filename = str(int(ID)).zfill(17)
            name = filename + ".npy"
            name = os.path.join(args.confidence_root, self.args.pretrained_2d_model, name)
            self.con_list[int(ID)] = name

        self.depth_list = {}
        for i, ID in enumerate(imageId):
            filename = str(int(ID)).zfill(4)
            name = filename + ".npy"
            name = os.path.join(args.depth_root, str(args.scene), name)
            self.depth_list[int(ID)] = name

        self.dynamic_mask_list = {}
        for i, ID in enumerate(imageId):
            filename = str(int(ID)).zfill(4)
            name = filename + ".npy"
            name = os.path.join(args.dymanic_mask_root, str(args.scene), name)
            self.dynamic_mask_list[int(ID)] = name

        pts_filename = os.path.join(self.args.points_root, str(self.args.scene) + ".las")
        las = laspy.read(pts_filename)
        all_points = np.stack((las.x, las.y, las.z), axis=1)
        self.center_pose = np.mean(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)
        min_xyz = np.min(all_points, axis=0)
        delta_xyz = max_xyz - min_xyz
        self.dist = np.linalg.norm(delta_xyz)

        self.translation = np.array(self.center_pose)
        
        self.metas = []
        self.build_metas()

    def build_metas(self):
        for ID in self.images_list.keys():
            if ID not in self.IDs: continue
            print(ID)

            image_path = self.images_list[ID] # rgb
            sem_path = self.sem_list[ID] # semantic
            con_path = self.con_list[ID] # confidence
            dynamic_mask_path = self.dynamic_mask_list[ID] # dynamic_mask
            
            image = (np.array(cv2.imread(image_path)) / 255.).astype(np.float32)
            sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
            confidence = np.load(con_path)
            dynamic_mask = np.load(dynamic_mask_path)

            pose = np.asarray(self.cam_infos[ID]).astype(np.float32)
            pose[:3] = pose[:3] - self.translation
            ray_dir = build_ray_dir(image.shape[0], image.shape[1])
            ray_dir = ray_dir.reshape(-1, 3)

            depth = np.load(self.depth_list[ID])
            depth = depth.reshape(-1)

            image = image.reshape(-1, 3) # N 3
            sem = sem.reshape(-1) # N
            confidence = confidence.reshape(-1) # N
            dynamic_mask = dynamic_mask.reshape(-1) # N

            sem = sem[self.car_mask]
            image = image[self.car_mask]
            confidence = confidence[self.car_mask]
            depth = depth[self.car_mask]
            ray_dir = ray_dir[self.car_mask]
            dynamic_mask = dynamic_mask[self.car_mask]

            sem = sem[dynamic_mask]
            image = image[dynamic_mask]
            confidence = confidence[dynamic_mask]
            depth = depth[dynamic_mask]
            ray_dir = ray_dir[dynamic_mask] 
            
            if self.args.use_sky == False:
                flag = (depth < self.args.depth_threhold) & (sem != 10) & (sem != 2)
                sem = sem[flag]
                image = image[flag]
                confidence = confidence[flag]
                depth = depth[flag]
                ray_dir = ray_dir[flag]

            sem = self.mapping_label(sem, self.args.class_num, self.label_list)
            mapper = np.vectorize(self.training_cityscape_to_new_whu3d.get)
            sem = mapper(sem)

            if image.shape[0] < self.args.N_rays:
                print(ID, image.shape[0])
                continue

            sem_uniq = set(sem)
            sem_count = {}
            for sem_index in sem_uniq:
                sem_count[sem_index] = (sem == sem_index).sum()

            N_rays = self.args.N_rays
            class_ray_num = np.asarray(list(sem_count.values()))
            class_ray_num = np.log(class_ray_num) / np.log(10)
            class_ray_num = np.exp(class_ray_num) / np.sum(np.exp(class_ray_num))
            class_ray_num = (N_rays * class_ray_num).astype(np.int32)
            difference = N_rays - class_ray_num.sum()
            if difference > 0:
                lenth = len(class_ray_num)
                index = np.random.choice(np.arange(lenth), size=difference, replace = True)
                np.add.at(class_ray_num, index, 1)
            for i, key in enumerate(sem_count.keys()):
                sem_count[key] = class_ray_num[i]
                
            self.metas.append((image, sem, ray_dir, depth, confidence, sem_count, pose))
    
    def build_ray_points(self, rays):       
        rays_o, rays_d = rays[:, :3], rays[:, 3:6] # N 3
        ray_dir = rays_d.copy() # N 3
        scale_factor = np.linalg.norm(rays_d, ord=2, axis=1)
        ray_dir = ray_dir / scale_factor[:, None]
        near, far, samples_all = self.args.near, self.args.far, self.args.samples_all
        delta = (far-near)/(samples_all-1)
        z_steps = np.linspace(0, 1, samples_all)
        z_vals = near * (1-z_steps) + far * z_steps
        xyz = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[None, :, None]/ scale_factor[:, None, None]
        #     N 1 3                 N 1 3               1 samples_all 1                           N 1 1
        return xyz, ray_dir 
        
    def __getitem__(self, index):
        image, sem, ray_dir, depth, confidence, sem_count, pose = self.metas[index]

        if self.split == "train":
            N_rays = self.args.N_rays

            if len(image) < N_rays:
                N_rays = len(image)

            rand_ids = []
            for sem_index in sem_count.keys():
                index = np.arange(len(sem))
                flag = (sem == sem_index) 
                index = index[flag]
                num = sem_count[sem_index]
                random_index = np.random.choice(index, size=num, replace=True)
                rand_ids.append(random_index)
            rand_ids = np.concatenate(rand_ids, 0)
            shuffle(rand_ids)

            image = image[rand_ids[:N_rays]]
            sem = sem[rand_ids[:N_rays]]
            ray_dir = ray_dir[rand_ids[:N_rays]]
            depth = depth[rand_ids[:N_rays]]
            confidence = confidence[rand_ids[:N_rays]]
            
        ret = {
            'rays_rgb': image.astype(np.float32), # 2048 3
            'pseudo_label': sem, # 2048
            'ray_dir': ray_dir.astype(np.float32),
            'depth': depth.astype(np.float32),
            'confidence': confidence.astype(np.float32),
            'center_pose': self.center_pose,
            'dist': self.dist,
            'pose': pose,
        }
            
        return ret
    
    def __len__(self):
        return len(self.metas)
    
    def mapping_label(self, sem, class_num=16, label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18]):
        mask = np.in1d(sem, label_list).reshape(sem.shape)
        mapping = np.searchsorted(label_list, sem)
        result = np.zeros_like(mapping)
        for i, l in enumerate(label_list):
            result[mapping == i] = i
        sem = result.copy()
        sem[~mask] = class_num - 1
        return sem