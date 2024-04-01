import os
import cv2
import sys
import glob
import torch
import warnings
import json
from fft_process import process_video as process_video_FFT, process_video_DCT, process_video_DWT, process_video_DST

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch.utils.data as data

sys.path.append("..")

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def read_csv(path):
    data = open(path).readlines()
    # names = data[0].replace('\n', '').split('|')
    names = ['path', 'gloss']
    save_arr = []
    for line in data:
        save_dict = {name: 0 for name in names}
        line = line.replace('\n', '').split('|')
        for name, item in zip(names, line):
            save_dict[name] = item
        save_arr.append(save_dict)
    return save_arr


class UCF(data.Dataset):

    def __init__(self, prefix, mode="train", clean=False, poison_ratio=0.2, downsample=32, F=None, X=None, Y=None,
                 F_lower=None, F_upper=None, pert=5e5, gt_path='../ucfTrainTestlist', method='FFT'):
        self.mode = mode
        self.prefix = prefix
        self.clean = clean
        self.pert = pert
        self.label_dict = {}

        if method == 'FFT':
            self.process_video = process_video_FFT
        elif method == 'DCT':
            self.process_video = process_video_DCT
        elif method == 'DWT':
            self.process_video = process_video_DWT
        elif method == 'DST':
            self.process_video = process_video_DST
        else:
            print('the transform method is not implemented yet')
            exit()

        train_gt_path = os.path.join(gt_path, 'train.json')
        test_gt_path = os.path.join(gt_path, 'test.json')

        train_input_list = json.load(open(train_gt_path))
        temp_test_input_list = json.load(open(test_gt_path))
        test_input_list = []
        val_input_list = []
        for i, path in enumerate(temp_test_input_list):
            if i % 10 == 0:
                val_input_list.append(path)
            else:
                test_input_list.append(path)

        # self.freq = downsample if downsample > 0 else 100
        self.downsample = downsample
        for i, c in enumerate(sorted(os.listdir(self.prefix))):
            self.label_dict[c] = i
        # input_list = sorted(glob.glob(os.path.join(self.prefix, '*', '*')))

        if self.mode == 'train':
            self.input_list = train_input_list
        elif self.mode == 'test':
            self.input_list = test_input_list
        else:  # 'val'
            assert self.clean
            self.input_list = val_input_list

        if F is not None and X is not None and Y is not None:
            self.F = F
            self.X = X
            self.Y = Y
        else:
            if F_upper is None and F_lower is None:
                F_lower, F_upper = 35, 45
            self.F = np.arange(F_lower, F_upper)
            self.X = [96, 72, 60, 149, 124, 57, 7, 66, 203, 140, 46, 97, 169,
                      21, 191, 196, 61, 95, 77, 184, 171, 75, 89, 218, 205]
            self.Y = [99, 2, 205, 40, 22, 7, 187, 70, 148, 177, 204, 77, 176,
                      120, 88, 156, 190, 81, 30, 93, 206, 10, 157, 48, 165]

        if self.mode == 'train':
            total_num = len(self.input_list)
            self.index = np.random.choice(np.arange(total_num), int(total_num * poison_ratio),
                                                       replace=False)
        else:
            self.index = list(range(len(self.input_list)))

        if not self.clean:
            print(f'using {method} transform for trigger embedding')
            print(f"perturbation size {self.pert}, num of perturbation {len(self.X)} {len(self.Y)}")
            print(f"poisoning ratio {len(self.index) / len(self.input_list)}, ")
            print(f"true freq {self.F}")
        else:
            print("no poisoning")

        self.transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        )

        print(mode, len(self.input_list))

    def __getitem__(self, idx):

        path = self.input_list[idx]
        c = path.split('/')[-2]
        label = self.label_dict[c]

        poison = idx in self.index

        if label == 0:
            poison = False
        if self.clean:
            poison = False
            fake_label = label
        else:
            if poison and label != 0:
                fake_label = 0
            else:
                fake_label = label

        if poison:
            F = self.F
            X = self.X
            Y = self.Y

        else:
            F, X, Y = None, None, None

        img_folder = os.path.join(self.prefix, path + '/*.jpg')

        img_list = sorted(glob.glob(img_folder))

        upsampled_indices = list(np.floor(np.linspace(0, len(img_list) - 1, self.downsample)).astype(int))
        img_list = [img_list[i] for i in upsampled_indices]

        imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

        if len(imgs) == 0:
            print(img_folder)

        if poison:

            pro_imgs = []
            imgs = np.stack(imgs)
            for i in range(3):
                img = imgs[:, :, :, i]
                pro_img = self.process_video(img, X, Y, F, self.pert)
                pro_imgs.append(pro_img)
            imgs = np.stack(pro_imgs)

        else:
            imgs = np.stack(imgs)
            imgs = np.transpose(imgs, (3, 0, 1, 2))  # l*h*w*c -> c*l*h*w
        imgs = torch.from_numpy(imgs)

        imgs = self.transform(imgs)

        return imgs[0], imgs[1], label, fake_label

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, fake_label = zip(*batch)
        label = torch.LongTensor(label)
        fake_label = torch.LongTensor(fake_label)

        placehoder = torch.ones(1)

        return video[0][0], video[0][1], placehoder, label, fake_label

    def __len__(self):
        return len(self.input_list) - 1
