import os
import cv2
import glob
import torch
import numpy
import random
import shutil

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

    




if __name__ == "__main__":
    '''
    KITTI Dataset의 경우 Train과 Valid가 나누어져 있지 않아 작업 필요
    python -m dataset.detection.utils
    '''
    kit = KITTI(mnt_path='/mnt', img_path='/image')
    kit.split_valid_set()