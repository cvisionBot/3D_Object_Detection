import os
import cv2
import glob
import torch
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.estimation.calibration import Calibration, draw_projected_box3d

class KITTI:
    def __init__(self, mnt_path, img_path, ratio=0.3):
        self.mnt_path = mnt_path
        self.img_path = img_path
        self.ratio = ratio

    def create_train_folder(self):
        if not(os.path.isdir(self.mnt_path + '/kit_train')):
            os.mkdir(self.mnt_path + '/kit_train')
        else:
            print('directory already exist!!')

    def create_valid_folder(self):
        if not(os.path.isdir(self.mnt_path + '/kit_valid')):
            os.mkdir(self.mnt_path+'/kit_valid')
        else:
            print('directory already exist!!')



    def split_valid_set(self):
        if not(os.path.exists(self.mnt_path + self.img_path)):
            print("Image Not Exist!!")
        imgs_file = glob.glob((self.mnt_path+self.img_path) + '/*.png')
        random.shuffle(imgs_file)
        self.create_train_folder()
        self.create_valid_folder()
        train_set = imgs_file[:int(len(imgs_file) * self.ratio)]
        valid_set = imgs_file[int(len(imgs_file) * self.ratio):]

        for i in train_set:
            annot = i.replace("png", "txt")
            shutil.copy(i, self.mnt_path + '/kit_train')
            shutil.copy(annot, self.mnt_path + '/kit_train')
        for i in valid_set:
            annot = i.replace("png", "txt")
            shutil.copy(i, self.mnt_path + '/kit_valid')
            shutil.copy(annot, self.mnt_path + '/kit_valid')
        return "Data split end!!"

    
class Visualize3D:
    def __init__(self, img_path, annot_path, calib_path):
        self.image = cv2.imread(img_path)
        self.annot_df = self.read_annotation(annot_path)
        self.calib_path = calib_path
        self.calib = Calibration(self.calib_path)
        self.names = self.parse_name('/home/insig/3D_Pose_Estimation/dataset/estimation/names/kitti.txt')
        self.color = self.gen_random_colors(self.names)

    def read_annotation(self, annot_path):
        df = pd.read_csv(annot_path, header=None, sep= ' ')
        df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
        # df = df[df['type'] == 'Car']
        df.reset_index(drop=True, inplace=True)
        return df

    def gen_random_colors(self, names):
        colors = [(random.randint(0, 255),
                   random.randint(0, 225),
                   random.randint(0, 255)) for i in range(len(names))]
        return colors

    def parse_name(self, name_file):
        with open(name_file, 'r') as f:
            return f.read().splitlines()

    def visualize(self):
        for raws in range(len(self.annot_df)):
            name = self.annot_df.iloc[raws]['type']
            corner_3d_cam2 = self.compute_3d_box_cam2(*self.annot_df.loc[raws, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
            pts_2d = self.calib.project_rect_to_image(corner_3d_cam2.T)
            self.image = draw_projected_box3d(self.image, pts_2d, color=self.color[self.names.index(str(name))], thickness=1)
        cv2.imwrite("visualize_image.png", self.image)
    
    def compute_3d_box_cam2(self, h, w, l, x, y, z, yaw):
        """
        Return : 3xn in cam2 coordinate
        """
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        corners_3d_cam2 += np.vstack([x, y, z])
        return corners_3d_cam2

if __name__ == "__main__":
    '''
    KITTI Dataset의 경우 Train과 Valid가 나누어져 있지 않아 작업 필요
    python -m dataset.detection.utils
    '''
    vis_3D = Visualize3D(img_path='/mnt/kit_train/000011.png', annot_path='/mnt/kit_train/000011.txt', calib_path='/mnt/kit_calib/000011.txt')
    vis_3D.visualize()