from torch.utils.data import Dataset
import numpy as np
import os 
import cv2
from src.utils.kitti360_utils import build_rays
from plyfile import PlyData

from random import shuffle

city_scapes_names = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]
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
    [119, 11, 32]
])

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
    
def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

class Kitti360Dataset(Dataset):

    def __init__(self, args, split="train"):
        """ Load data from given dataset directory. """
        self.split = split
        self.args = args
        
        start, end, pad_nums = [int(args.start), int(args.end), int(args.pad_nums)]
        self.IDs = range(start - pad_nums, end + pad_nums + 1, args.train_interval)
        if self.split == "val" or self.split == "test":
            self.IDs = [start]

        self.label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18]
        self.kitti360_to_city_scapes = {
                            0: 19,
                            1: 13,
                            2: 19,
                            3: 19,
                            4: 19,
                            5: 19,
                            6: 1,
                            7: 0,
                            8: 1,
                            9: 1,
                            10: 19,
                            11: 2,
                            12: 3,
                            13: 4,
                            14: 4,
                            15: 19,
                            16: 19,
                            17: 5,
                            18: 5,
                            19: 6,
                            20: 7,
                            21: 8,
                            22: 9,
                            23: 10,
                            24: 11,
                            25: 12,
                            26: 13,
                            27: 14,
                            28: 16,
                            29: 15,
                            30: 14,
                            31: 14,
                            32: 17,
                            33: 18,
                            34: 2,
                            35: 2,
                            36: 7,
                            37: 5,
                            38: 5, 
                            39: 19,
                            40: 19,
                            41: 19,
                            42: 2,
                            43: 13,
                            44: 19,
                            -1: 19,
                        }
        
        self.city_scapes_to_new_kitti360 = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 10,
            7: 6,
            8: 7,
            9: 8,
            10: 10,
            11: 10,
            12: 10,
            13: 9,
            14: 10,
            15: 10,
            16: 10,
            17: 10,
            18: 10,
            19: 10
        }

        calib_dir = os.path.join(args.data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.K = self.K_00[:, :-1]

        self.fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        TrCamToPose = self.loadCalibrationCameraToPose(self.fileCameraToPose)
        self.TrCam1ToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose['image_01']

        self.cam2world_dic = {}
        for line in open(args.cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dic[int(value[0])] = np.array(value[1:]).reshape(4, 4)

        images_list_00 = {}
        images_list_01 = {}
        self.imageId = [int(filename[:-4]) for filename in os.listdir(os.path.join(args.images_root, "image_00", "data_rect"))]
        self.imageId = sorted(self.imageId)
        for i, ID in enumerate(self.imageId):
            filename = str(int(ID)).zfill(10)
            name = os.path.join(args.images_root, "image_00", "data_rect", filename + ".png")
            images_list_00[int(ID)] = name
            name = os.path.join(args.images_root, "image_01", "data_rect", filename + ".png")
            images_list_01[int(ID)] = name
        self.images_list = [images_list_00, images_list_01]

        sem_list_00 = {}
        sem_list_01 = {}
        if self.args.use_2d_gt:
            for i, ID in enumerate(self.imageId):
                filename = str(int(ID)).zfill(10)
                name = os.path.join(args.gt_2d_sem_root, "image_00", "semantic", filename + ".png")
                sem_list_00[int(ID)] = name
                name = os.path.join(args.gt_2d_sem_root, "image_01", "semantic", filename + ".png")
                sem_list_01[int(ID)] = name
        else:
            for i, ID in enumerate(self.imageId):
                filename = str(int(ID)).zfill(10)
                name = os.path.join(args.sem_root, "image_00", "semantic", self.args.pretrained_2d_model, filename + ".png")
                sem_list_00[int(ID)] = name
                name = os.path.join(args.sem_root, "image_01", "semantic", self.args.pretrained_2d_model, filename + ".png")
                sem_list_01[int(ID)] = name
        self.sem_list = [sem_list_00, sem_list_01]

        con_list_00 = {}
        con_list_01 = {}
        if self.args.use_2d_gt:
            for i, ID in enumerate(self.imageId):
                filename = str(int(ID)).zfill(10)
                name = os.path.join(args.gt_2d_sem_root, "image_00", "confidence", filename + ".png")
                con_list_00[int(ID)] = name
                name = os.path.join(args.gt_2d_sem_root, "image_01", "confidence", filename + ".png")
                con_list_01[int(ID)] = name
        else:
            for i, ID in enumerate(self.imageId):
                filename = str(int(ID)).zfill(10)
                name = os.path.join(args.confidence_root, "image_00", "confidence", self.args.pretrained_2d_model, filename + ".npy")
                con_list_00[int(ID)] = name
                name = os.path.join(args.confidence_root, "image_01", "confidence", self.args.pretrained_2d_model, filename + ".npy")
                con_list_01[int(ID)] = name
        self.con_list = [con_list_00, con_list_01]

        depth_list_00 = {}
        depth_list_01 = {}
        for i, ID in enumerate(self.imageId):
            filename = str(int(ID)).zfill(10)
            name = os.path.join(args.depth_root, "image_00", "depth_r_ip_basic", filename + ".npy")
            depth_list_00[int(ID)] = name
            name = os.path.join(args.depth_root, "image_01", "depth_r_ip_basic", filename + ".npy")
            depth_list_01[int(ID)] = name
        self.depth_list = [depth_list_00, depth_list_01]

        points_filename = str(start).zfill(10) + "_" + str(end).zfill(10) + ".ply"
        points_filename = os.path.join(args.points_root, points_filename)
        all_points, _, _ = read_ply(points_filename)
        
        self.center_pose = np.mean(all_points, axis=0)
        max_xyz = np.max(all_points, axis=0)
        min_xyz = np.min(all_points, axis=0)
        delta_xyz = max_xyz - min_xyz
        self.dist = np.linalg.norm(delta_xyz)

        self.translation = np.array(self.center_pose)
        
        self.metas = []
        self.build_metas()

    def checkfile(self, filename):
        if not os.path.isfile(filename):
            raise RuntimeError('%s does not exist!' % filename)
    
    def readVariable(self, fid,name,M,N):
        # rewind
        fid.seek(0,0)
        
        # search for variable identifier
        line = 1
        success = 0
        while line:
            line = fid.readline()
            if line.startswith(name):
                success = 1
                break

        # return if variable identifier not found
        if success==0:
            return None
        
        # fill matrix
        line = line.replace('%s:' % name, '')
        line = line.split()
        assert(len(line) == M*N)
        line = [float(x) for x in line]
        mat = np.array(line).reshape(M, N)

        return mat

    def loadCalibrationCameraToPose(self, filename):
        # check file
        self.checkfile(filename)

        # open file
        fid = open(filename,'r');
            
        # read variables
        Tr = {}
        cameras = ['image_00', 'image_01', 'image_02', 'image_03']
        lastrow = np.array([0,0,0,1]).reshape(1,4)
        for camera in cameras:
            Tr[camera] = np.concatenate((self.readVariable(fid, camera, 3, 4), lastrow))
            
        # close file
        fid.close()
        return Tr

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect
        
    def build_metas(self):
        for ID in self.cam2world_dic.keys():
            if ID not in self.IDs: continue
            print(ID)
            
            if self.args.use_1th_cam:
                cams = 2
            else:
                cams = 1
            for index in range(0, cams):
                image_path = self.images_list[index][ID] # rgb
                sem_path = self.sem_list[index][ID] # semantic
                con_path = self.con_list[index][ID] # confidence

                depth = np.load(self.depth_list[index][ID])
                depth = depth.reshape(-1)
                
                if not os.path.exists(sem_path): continue
                
                image = (np.array(cv2.imread(image_path)) / 255.).astype(np.float32)
                sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
                
                confidence = np.load(con_path)

                pose = self.cam2world_dic[ID].copy()
                pose[:3, 3] = pose[:3, 3] - self.translation
                ray = build_rays(self.K, pose, image.shape[0], image.shape[1], self.TrCam1ToCam0, index) # N 6

                image = image.reshape(-1, 3) # N 3
                sem = sem.reshape(-1) # N 3
                confidence = confidence.reshape(-1)

                flag = (depth < self.args.depth_threhold) & (sem != 10) 
                sem = sem[flag]
                image = image[flag]
                confidence = confidence[flag]
                depth = depth[flag]
                ray = ray[flag]

                if self.args.class_num == 11:
                    mapper = np.vectorize(self.city_scapes_to_new_kitti360.get)
                    sem = mapper(sem)
                    print("done sem trans 11")
                else:
                    sem = self.mapping_label(sem, self.args.class_num, self.label_list)
                    print("done sem trans 16")

                if image.shape[0] < self.args.N_rays:
                    print(ID, image.shape[0])
                    continue

                sem_uniq = set(sem)
                sem_count = {}
                for sem_index in sem_uniq:
                    sem_count[sem_index] = (sem == sem_index).sum()
                self.metas.append((image, sem, ray, depth, confidence, sem_count))
        
    def build_ray_points(self, rays):       
        rays_o, rays_d = rays[:, :3], rays[:, 3:6] # N 3
        ray_dir = rays_d.copy() # N 3
        scale_factor = np.linalg.norm(rays_d, ord=2, axis=1)
        ray_dir = ray_dir / scale_factor[:, None]
        near, far, samples_all = self.args.near, self.args.far, self.args.samples_all
        z_steps = np.linspace(0, 1, samples_all)
        z_vals = near * (1-z_steps) + far * z_steps       
        xyz = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[None, :, None]/ scale_factor[:, None, None]
        #     N 1 3                 N 1 3               1 samples_all 1                           N 1 1
        return xyz, ray_dir   
        
    def __getitem__(self, index):
        image, sem, ray, depth, confidence, sem_count = self.metas[index]
        
        if self.split == "train":
            N_rays = self.args.N_rays

            if len(image) < N_rays:
                N_rays = len(image)
                
            N_rays = self.args.N_rays
            class_ray_num = [N_rays // len(sem_count) for i in range(len(sem_count)-1)]
            class_ray_num.append(N_rays - (N_rays // len(sem_count))*(len(sem_count)-1))

            rand_ids = []
            for i in range(len(sem_count)):
                sem_index = list(sem_count.keys())[i]
                index = np.arange(len(sem))
                flag = (sem == sem_index) 
                index = index[flag]
                num = class_ray_num[i] 
                random_index = np.random.choice(index, size=num, replace=True)
                rand_ids.append(random_index)
            rand_ids = np.concatenate(rand_ids, 0)
            shuffle(rand_ids) 
            
            image = image[rand_ids[:N_rays]]
            sem = sem[rand_ids[:N_rays]]
            ray = ray[rand_ids[:N_rays]]
            depth = depth[rand_ids[:N_rays]]
            confidence = confidence[rand_ids[:N_rays]]

        ret = {
            'rays_rgb': image.astype(np.float32), # 2048 3
            'pseudo_label': sem, # 2048
            'ray': ray.astype(np.float32),
            'depth': depth.astype(np.float32),
            'confidence': confidence.astype(np.float32),
            'center_pose': self.center_pose,
            'dist': self.dist,
        }
            
        return ret
    
    def __len__(self):
        return len(self.metas)
    
    def get_random_points(self, points, rgb, feat, points_num, num=512):
        indices = np.random.choice(points_num, num, replace=False)
        return points[indices], rgb[indices], feat[indices]
    
    def mapping_label(self, sem, class_num=16, label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 17, 18]):
        mask = np.in1d(sem, label_list).reshape(sem.shape)
        mapping = np.searchsorted(label_list, sem)
        result = np.zeros_like(mapping)
        for i, l in enumerate(label_list):
            result[mapping == i] = i
        sem = result.copy()
        sem[~mask] = class_num - 1
        return sem