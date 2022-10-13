import cv2
import glob
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

class KITTIDataset(Dataset):
    def __init__(self, path=None, is_train=True):
        super(KITTIDataset, self).__init__()
        if is_train:
            self.image = glob.glob(path +'/kit_train/*.png')
        else:
            self.image = glob.glob(path + '/kit_valid/*.png')
        
        with open("/names/kitti.txt") as f:
            class_list = f.readline()
        self.class_list = [class_list.rstrip('\n') for line in class_list]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_file = self.image[index]
        img = cv2.imread(image_file)
        annot = self.load_annotations(img, img.shape)
        # annot = function
        # transformes = transformes(image, annot) function
        return {'image':img} # return transformes[image, annot]

    def load_annotations(self, img_file, img_shape):
        img_h, img_h, _ = img_shape
        annotations_file = img_file.replace('.png', '.txt')
        

if __name__ == "__main__":
    '''
    Data Loader 테스트 코드
    python -m dataset.detection.kitti_format
    '''

    loader = DataLoader(KITTIDataset(path='/mnt/'), batch_size=1, shuffle=True)

    for batch, sample in enumerate(loader):
        imgs = sample['image']
        visualize(imgs)