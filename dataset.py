import os
import glob
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import PIL

from PIL import Image, ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

def same_or_not(percent):
    return random.randrange(100) < percent


def color_masking(img, r, g, b):
    return np.logical_and(np.logical_and(img[:, :, 0] == r, img[:, :, 1] == g), img[:, :, 2] == b)


def logical_or_masks(mask_list):
    mask_all = np.zeros_like(mask_list[0], dtype=bool)
    for mask in mask_list:
        mask_all = np.logical_or(mask_all, mask)
    return mask_all


def parsing2mask(paring):
    img_numpy = np.array(paring)

    mask_nose = color_masking(img_numpy, 76, 153, 0)
    mask_left_eye = color_masking(img_numpy, 204, 0, 204)
    mask_right_eye = color_masking(img_numpy, 51, 51, 255)
    mask_skin = color_masking(img_numpy, 204, 0, 0)
    mask_left_eyebrow = color_masking(img_numpy, 255, 204, 204)
    mask_right_eyebrow = color_masking(img_numpy, 0, 255, 255)
    mask_up_lip = color_masking(img_numpy, 255, 255, 0)
    mask_mouth_inside = color_masking(img_numpy, 102, 204, 0)
    mask_down_lip = color_masking(img_numpy, 0, 0, 153)
    mask_left_ear = color_masking(img_numpy, 255, 0, 0)
    mask_right_ear = color_masking(img_numpy, 102, 51, 0)

    mask_face = logical_or_masks(
        [mask_nose, mask_left_eye, mask_right_eye, mask_skin, mask_left_eyebrow, mask_right_eyebrow, mask_up_lip,
         mask_mouth_inside, mask_down_lip, mask_left_ear, mask_right_ear, ])
    mask_face = 1.0 * mask_face
    mask_face = Image.fromarray(np.array(mask_face))
    return mask_face


class HifiFaceParsingTrainDataset(Dataset):
    def __init__(self, img_root, parsing_root, same_rate=50, transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                         transforms.CenterCrop((256, 256)),
                                                                         transforms.ToTensor()])):
        super(HifiFaceParsingTrainDataset, self).__init__()
        ext_list = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

        self.img_root = img_root
        img_dir = Path(img_root)

        img_files = []
        for ext in ext_list:
            file_generator = img_dir.glob(f"**/*.{ext}")
            img_files.extend([file for file in file_generator])
        img_files.sort()
        self.img_files = img_files

        self.parsing_root = parsing_root
        parsing_dir = Path(parsing_root)
        parsing_files = []
        for ext in ext_list:
            file_generator = parsing_dir.glob(f"**/*.{ext}")
            parsing_files.extend([file for file in file_generator])
        parsing_files.sort()
        self.parsing_files = parsing_files

        assert len(self.img_files) == len(self.parsing_files), f"number of image files and parsing files are different. ({len(self.img_files)} and {len(self.parsing_files)})"
        for img_path, parsing_path in zip(self.img_files, self.parsing_files):
            _, img_path_ = os.path.split(img_path)
            _, parsing_path_ = os.path.split(parsing_path)
            img_path_, ext = os.path.splitext(img_path_)
            parsing_path_, ext = os.path.splitext(parsing_path_)
            assert img_path_ == parsing_path_, f"image file and parsing file not matched, {img_path}, {parsing_path}"

        self.same_rate = same_rate
        self.transform = transform


    def __getitem__(self, index):
        l = self.__len__()
        s_idx = index
        if same_or_not(self.same_rate):
            t_idx = s_idx
        else:
            t_idx = random.randrange(l)

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.img_files[t_idx])
        s_img = Image.open(self.img_files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        f_parsing = Image.open(self.parsing_files[t_idx])
        f_parsing = f_parsing.convert('RGB')
        f_mask = parsing2mask(f_parsing)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = self.transform(f_mask)

        return {'target_image': f_img,
                'source_image': s_img,
                'target_mask': f_mask,
                'same': same,
                }

    def __len__(self):
        return len(self.img_files)

class HifiFaceParsingTrainRecDataset(Dataset):
    def __init__(self, img_root, parsing_root, same_rate=50, transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                         transforms.CenterCrop((256, 256)),
                                                                         transforms.ToTensor()])):
        super(HifiFaceParsingTrainRecDataset, self).__init__()
        ext_list = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

        self.img_root = img_root
        img_dir = Path(img_root)

        img_files = []
        for ext in ext_list:
            file_generator = img_dir.glob(f"**/*.{ext}")
            img_files.extend([file for file in file_generator])
        img_files.sort()
        self.img_files = img_files

        img_file_dict = {}
        for idx, img_file in enumerate(img_files):
            dir_name, file_name = os.path.split(img_file)
            if not str(dir_name) in img_file_dict:
                img_file_dict[str(dir_name)] = []
            img_file_dict[str(dir_name)].append(idx)
        self.img_file_dict = img_file_dict

        self.parsing_root = parsing_root
        parsing_dir = Path(parsing_root)
        parsing_files = []
        for ext in ext_list:
            file_generator = parsing_dir.glob(f"**/*.{ext}")
            parsing_files.extend([file for file in file_generator])
        parsing_files.sort()
        self.parsing_files = parsing_files

        assert len(self.img_files) == len(
            self.parsing_files), f"number of image files and parsing files are different. ({len(self.img_files)} and {len(self.parsing_files)})"
        for img_path, parsing_path in zip(self.img_files, self.parsing_files):
            _, img_path_ = os.path.split(img_path)
            _, parsing_path_ = os.path.split(parsing_path)
            img_path_, ext = os.path.splitext(img_path_)
            parsing_path_, ext = os.path.splitext(parsing_path_)
            assert img_path_ == parsing_path_, f"image file and parsing file not matched, {img_path}, {parsing_path}"

        self.same_rate = same_rate
        self.transform = transform

    def __getitem__(self, index):
        l = self.__len__()
        s_idx = index
        if same_or_not(self.same_rate):
            file_path = self.img_files[s_idx]
            dir_name, _ = os.path.split(file_path)
            same_identity_idx = random.randrange(len(self.img_file_dict[str(dir_name)]))
            t_idx = self.img_file_dict[str(dir_name)][same_identity_idx]
        else:
            t_idx = random.randrange(l)

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.img_files[t_idx])
        s_img = Image.open(self.img_files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        f_parsing = Image.open(self.parsing_files[t_idx])
        f_parsing = f_parsing.convert('RGB')
        f_mask = parsing2mask(f_parsing)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = self.transform(f_mask)

        return {'target_image': f_img,
                'source_image': s_img,
                'target_mask': f_mask,
                'same': same,
                }

    def __len__(self):
        return len(self.img_files)

class HifiFaceParsingValDataset(Dataset):
    def __init__(self, img_root, parsing_root, transform=transforms.Compose([transforms.Resize((256, 256)),
                                                           transforms.CenterCrop((256, 256)),
                                                           transforms.ToTensor()])):
        super(HifiFaceParsingValDataset, self).__init__()
        ext_list = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

        self.img_root = img_root
        img_dir = Path(img_root)

        img_files = []
        for ext in ext_list:
            file_generator = img_dir.glob(f"**/*.{ext}")
            img_files.extend([file for file in file_generator])
        img_files.sort()
        self.img_files = img_files

        self.parsing_root = parsing_root
        parsing_dir = Path(parsing_root)
        parsing_files = []
        for ext in ext_list:
            file_generator = parsing_dir.glob(f"**/*.{ext}")
            parsing_files.extend([file for file in file_generator])
        parsing_files.sort()
        self.parsing_files = parsing_files

        assert len(self.img_files) == len(self.parsing_files), f"number of image files and parsing files are different. ({len(self.img_files)} and {len(self.parsing_files)})"
        for img_path, parsing_path in zip(self.img_files, self.parsing_files):
            _, img_path_ = os.path.split(img_path)
            _, parsing_path_ = os.path.split(parsing_path)
            img_path_, ext = os.path.splitext(img_path_)
            parsing_path_, ext = os.path.splitext(parsing_path_)
            assert img_path_ == parsing_path_, f"image file and parsing file not matched, {img_path}, {parsing_path}"

        self.transform = transform

    def __getitem__(self, index):
        l = len(self.img_files)

        t_idx = index // l
        s_idx = index % l

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.img_files[t_idx])
        s_img = Image.open(self.img_files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        f_parsing = Image.open(self.parsing_files[t_idx])
        f_parsing = f_parsing.convert('RGB')
        f_mask = parsing2mask(f_parsing)

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
            f_mask = self.transform(f_mask)

        return {'target_image': f_img,
                'source_image': s_img,
                'target_mask': f_mask,
                'same': same,
                }

    def __len__(self):
        return len(self.img_files) * len(self.img_files)
