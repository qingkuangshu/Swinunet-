# -*- coding: utf-8 -*-
import os
import cv2
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
 
 
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label
 
 
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
 
 
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
 
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
 
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
 
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
 
    def __len__(self):
        return len(self.sample_list)
 
    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
 
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
 
# 这里开始是自己添加的用于加载自己数据的标准数据加载函数，可以参考用于别的复现！
 
#***********************数据增强部分************************************
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
 
    return image
 
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
 
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
 
        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
 
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
 
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
 
    return image, mask
 
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
 
    return image, mask
 
def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
 
    return image, mask
 
def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
 
    return image, mask
 
#**********************加载自己数据相关的函数****************************
#遍历数据文件夹，这里注意路径是拼接的
def read_own_data(root_path, mode = 'train'):
    images = []
    masks = []
 
    image_root = os.path.join(root_path, mode + '/images')
    gt_root = os.path.join(root_path, mode + '/labels')
 
 
    for image_name in os.listdir(gt_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)
 
        images.append(image_path)
        masks.append(label_path)
 
    return images, masks
 
#训练数据读取
def own_data_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (512,512), interpolation = cv2.INTER_NEAREST)
    mask = cv2.imread(mask_path, 0)
    # mask = cv2.resize(mask, (512,512), interpolation = cv2.INTER_NEAREST)
 
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
 
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
 
    mask = np.expand_dims(mask, axis=2)
 
    img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
    # img = np.array(img, np.float32) / 255.0
    # mask = np.array(mask, np.float32)
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
 
    img = np.array(img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
    return img, mask
 
#验证数据读取
def own_data_test_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (512,512), interpolation = cv2.INTER_NEAREST)
    mask = cv2.imread(mask_path, 0)
    # mask = cv2.resize(mask, (512,512), interpolation = cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=2)
 
    img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
    # img = np.array(img, np.float32) / 255.0
    # mask = np.array(mask, np.float32)
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
 
    # mask[mask > 0] = 1
 
    img = np.array(img, np.float32).transpose(2, 0, 1)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)
 
    return img, mask
 
class ImageFolder(Dataset):
    def __init__(self, root_path, mode='train'):
        self.root = root_path
        self.mode = mode
        self.images, self.labels = read_own_data(self.root, self.mode)
 
    def __getitem__(self, index):
        if self.mode == 'test':
            img, mask = own_data_test_loader(self.images[index], self.labels[index])
        else:
            img, mask = own_data_loader(self.images[index], self.labels[index])
            img = torch.Tensor(img)
            mask = torch.Tensor(mask)
        return img, mask
 
    def __len__(self):
        # assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)