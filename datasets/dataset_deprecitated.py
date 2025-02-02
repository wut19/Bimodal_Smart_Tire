import glob
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt

class VTDataset(Dataset):
    """
    data strcuture:
        - data_dir
            - visual
                - modality_1(normal)
                    - class_1
                    - class_2
                    - ...
                - modality_2(smoky)
                ...
            - tactile
                - class_1
                - class_2
                - ...

    This structure is easy to be generalize to more visual and tactile modalities.
    
    NOTICE: this is a prior version, which is depreciated. Use MMVTDataset instead.
    """
    def __init__(self, data_dir=None, used_visual_modalities=[], random_visual=True, use_tactile=False, size=128, crop_size=300, is_test=False):
        self.data_dir = data_dir
        self.used_visual_modalities = used_visual_modalities
        self.use_tactile = use_tactile
        self.random_visual = random_visual
        self.is_test = is_test
        assert len(used_visual_modalities)>0 or use_tactile, 'You need at least one modality!!!'
        
        self.transform_visual = transforms.Compose([
            transforms.Resize([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.transform_tactile = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize([size, size]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.6902, 0.7033, 0.7269), std=(0.4, 0.4, 0.4))
        ])

        self.visual_dir = os.path.join(data_dir, 'visual')
        self.visual_modalities = os.listdir(self.visual_dir)
        for used_visual_modality in used_visual_modalities:
            assert used_visual_modality in self.visual_modalities, f'{used_visual_modalities} is not available!!!'
        self.visuals = {}
        for visual_modality in self.visual_modalities:
            dir_path = os.path.join(self.visual_dir, visual_modality)
            self.visuals[visual_modality] = glob.glob(f'{dir_path}/*/*.png')

        tactile_dir = os.path.join(data_dir, 'tactile')
        self.tactiles = glob.glob(f'{tactile_dir}/*/*.png')

        self.labels = [int(dir) for dir in os.listdir(tactile_dir)]

    def __getitem__(self, index):
        index_path = self.tactiles[index]
        dir, file_name = os.path.split(index_path)
        if self.is_test:
            print('-'*20)
        if len(self.used_visual_modalities) > 0:
            if self.random_visual:
                modal_name = self.used_visual_modalities[np.random.randint(len(self.used_visual_modalities))]
                visual_path = os.path.join(self.visual_dir, modal_name, dir.split('/')[-1], file_name)
                if self.is_test:
                    print(visual_path)
                visuals = Image.open(visual_path)
                visuals = self.transform_visual(visuals)
                visuals = visuals.unsqueeze(0)
            else:
                visuals = []
                for i, visual_modality in enumerate(self.used_visual_modalities):
                    visual_path = os.path.join(self.visual_dir, visual_modality, dir.split('/')[-1], file_name)
                    if self.is_test:
                        print(visual_path)
                    visuals.append(Image.open(visual_path))
                    visuals[-1] = self.transform_visual(visuals[-1])
                visuals = torch.stack(visuals, 0)
        else:
            visuals = 0

        if self.use_tactile:
            if self.is_test:
                print(index_path)
            tactile = Image.open(index_path)
            tactile = self.transform_tactile(tactile)
            tactile = torch.clip(tactile, 0, 1)
        else:
            tactile = 0
        
        label = int(dir.split('/')[-1]) - 1
        label = F.one_hot(torch.tensor([label]), num_classes=len(self.labels)).squeeze(0)
        if self.is_test:
            print(label)
            print('-'*20)

        data = {
            'visual': visuals,
            'tactile': tactile,
            'label': label,
        }
        return data

    def __len__(self):
        return len(self.tactiles)
    
def split_data(dataset, args):
    indices = list(range(len(dataset)))
    split = int(len(dataset) * args.data_split)
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )

    return train_data, val_data