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

class MMVTDataset(Dataset):
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
                - modality_1(tactile)
                    - class_1
                    - class_2
                    - ...
                - modality_2(inside visual)
                - ...
    This structure is easy to be generalized to more visual and tactile modalities.

    TODO: add meta information under data_dir/, that is
            - data_dir
                - ...
                - ...
                - metainfo.yaml/txt
    """
    def __init__(self, 
                 data_dir: str = None,
                 visual_mod_trans_mapping: dict = {},
                 random_visual: bool = True,
                 tactile_mod_trans_mapping: dict = {},
                 size: int = 128,
                 is_test: bool = False
    ):
        self.data_dir = data_dir
        self.visual_mod_trans_mapping = visual_mod_trans_mapping
        self.tactile_mod_trans_mapping = tactile_mod_trans_mapping
        self.used_visual_modalities = list(visual_mod_trans_mapping.keys())
        self.used_tactile_modalities = list(tactile_mod_trans_mapping.keys())
        self.random_visual = random_visual
        self.is_test = is_test
        assert len(self.used_visual_modalities)>0 or len(self.used_tactile_modalities), 'You need at least one modality!!!'   

        ## Notice that different modality needs different preprocessing method
        ## Modify below to customize your own dataset
        # TODO: add this part into configuration file
        self.transforms = {
            'v': transforms.Compose([
                    transforms.Resize([size, size]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]),
            't': transforms.Compose([
                    transforms.CenterCrop(300),   # used when segmented and processing tactile part independently
                    transforms.Resize([size, size]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.6902, 0.7033, 0.7269), std=(0.4, 0.4, 0.4))
                ]),
            't1': transforms.Compose([
                    transforms.CenterCrop(300),   # used when segmented and processing tactile part independently
                    transforms.Resize([size, size]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]),
            'vt': transforms.Compose([
                    transforms.Resize([size, size]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]),
        }

        self.visual_dir = os.path.join(data_dir, 'visual')
        self.visual_modalities = os.listdir(self.visual_dir)
        for used_visual_modality in self.used_visual_modalities:
            assert used_visual_modality in self.visual_modalities, f'{used_visual_modality} is not available!!!'
        self.visuals = {}
        for visual_modality in self.visual_modalities:
            dir_path = os.path.join(self.visual_dir, visual_modality)
            self.visuals[visual_modality] = glob.glob(f'{dir_path}/*/*.png')

        self.tactile_dir = os.path.join(data_dir, 'tactile')
        self.tactile_modalities = os.listdir(self.tactile_dir)
        for used_tactile_modality in self.used_tactile_modalities:
            assert used_tactile_modality in self.tactile_modalities, f'{used_tactile_modality} is not available!!!'
        self.tactiles = {}
        for tactile_modality in self.tactile_modalities:
            dir_path = os.path.join(self.tactile_dir, tactile_modality)
            self.tactiles[tactile_modality] = glob.glob(f'{dir_path}/*/*.png')

        # TODO: all the initialization should be completed by metainfo
        self.labels = [int(dir) for dir in os.listdir(os.path.join(self.tactile_dir, self.tactile_modalities[0]))]

    def __getitem__(self, index):

        # TODO: sample method should be obtained from metainfo
        index_path = self.tactiles[self.tactile_modalities[0]][index]
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
                transform_key = self.visual_mod_trans_mapping[modal_name]
                visuals = self.transforms[transform_key](visuals)
                visuals = visuals.unsqueeze(0)
            else:
                visuals = []
                for i, visual_modality in enumerate(self.used_visual_modalities):
                    visual_path = os.path.join(self.visual_dir, visual_modality, dir.split('/')[-1], file_name)
                    if self.is_test:
                        print(visual_path)
                    visuals.append(Image.open(visual_path))
                    transform_key = self.visual_mod_trans_mapping[visual_modality]
                    visuals[-1] = self.transforms[transform_key](visuals[-1])
                visuals = torch.stack(visuals, 0)
        else:
            visuals = 0

        if len(self.used_tactile_modalities) > 0:
            tactiles = []
            for i, tactile_modality in enumerate(self.used_tactile_modalities):
                tactile_path = os.path.join(self.tactile_dir, tactile_modality, dir.split('/')[-1], file_name)
                if self.is_test:
                    print(tactile_path)
                tactiles.append(Image.open(tactile_path))
                transform_key = self.tactile_mod_trans_mapping[tactile_modality]
                if self.is_test:
                    print(transform_key)
                tactiles[-1] = self.transforms[transform_key](tactiles[-1])
            tactiles = torch.stack(tactiles, 0)
            # tactiles = torch.clip(tactiles, 0, 1)
        else:
            tactiles = 0
        
        label = int(dir.split('/')[-1]) - 1
        label = F.one_hot(torch.tensor([label]), num_classes=len(self.labels)).squeeze(0)
        if self.is_test:
            print(label)
            print('-'*20)

        data = {
            'visual': visuals,
            'tactile': tactiles,
            'label': label,
        }
        return data

    def __len__(self):
        return len(self.tactiles[self.tactile_modalities[0]])
    
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

def test_parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--data_split', type=float, default=0.8, help='split data')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--random_seed', type=int, default=123, help='random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataset workers')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """ test """
    vt_dataset = MMVTDataset(
        data_dir='VisualTactileData_segT',
        visual_mod_trans_mapping={},
        random_visual=True,
        tactile_mod_trans_mapping={'t':'t'},
        is_test=True,
        )
    print(len(vt_dataset))
    # for key, value in vt_dataset[0].items():
    #     print(key, value.shape if isinstance(value, torch.Tensor) else value)

    # plt.imshow(vt_dataset[122]['tactile'][0].permute(1,2,0))
    # plt.show()

    args = test_parse_args()
    train_data, val_data = split_data(vt_dataset, args)

    for batch in train_data:
        print(batch['visual'].shape)
        print(batch['tactile'].shape)
        print(batch['label'].shape)