import os
import numpy as np
import torch
import pathlib
from pathlib import Path
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image

class LLVIR(Dataset):
    """LLVIR dataset."""

    def __init__(self, root_dir:str, train:bool, transforms=None, augmentations=None):
        """
        Arguments:
        ----------


        """
        
        root_dir = root_dir
        ann_file = os.path.join(root_dir, 'LLVIP.json')
        self.train = 'train' if train else 'test'
        self.rgbimg_dir = Path(root_dir) / 'visible'  / self.train
        self.irimg_dir = Path(root_dir) / 'infrared' / self.train
        print(f'RGB images directory: {self.rgbimg_dir}')
        print(f'IR images directory: {self.irimg_dir}')

        self.test_files = os.listdir(Path(root_dir) / 'visible' / 'test')
        self.coco = COCO(ann_file)
        self.ids = self.coco.getImgIds()
        
        self.transforms = transforms
        self.augmentation = augmentations

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]
        img_obj = self.coco.loadImgs([img_id])
        anns_obj = self.coco.loadAnns(self.coco.getAnnIds([img_id]))  

        if img_obj[0]['file_name'] in self.test_files and self.train == 'train':
            return self.__getitem__(np.random.randint(0, len(self.ids)))
        elif img_obj[0]['file_name'] not in self.test_files and self.train == 'test':
            return self.__getitem__(np.random.randint(0, len(self.ids)))

        if len(anns_obj) == 0:
            print(f'Image {idx} has no annotations.')

        # Read image
        rgb_img = Image.open(Path(self.rgbimg_dir) / img_obj[0]['file_name'])
        ir_img = Image.open(Path(self.irimg_dir) / img_obj[0]['file_name'])

        ## Make annotations lists
        bboxes = [ann['bbox'] for ann in anns_obj]
        # Format bbox from [x0,y0, w, h] to [x_center, y_center, w, h]
        bboxes = [[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, bbox[2], bbox[3]] for bbox in bboxes]
        areas = [ann['area'] for ann in anns_obj]
        labels = [ann['category_id'] for ann in anns_obj]

        # Make tensors
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(areas)
        iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)

        # Build target dictionary
        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            rgb_img = self.transforms(rgb_img)
            ir_img = self.transforms(ir_img)
        
        img = torch.cat((rgb_img, ir_img), dim=0)
        
        return img, target
        
class LLVIRDataModule(pl.LightningDataModule):
    """
    Data module for LLVIR

    Arguments:
    ----------
    root_dir: str
        Path to the root directory of the dataset
    rgb: bool
        If True then use RGB input, if False then use multispectral input
    batch_size: int
        Batch size
    num_workers: int
        Number of workers for the dataloader
    """
    def __init__(self, root_dir, train:bool, batch_size:int=1, num_workers:int=4):
        super().__init__()

        self.save_hyperparameters()
        self.annotations_path = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train

    def setup(self, stage=None):
        train_dataset = LLVIR(self.annotations_path, train=self.train, transforms=transforms.Compose([transforms.ToTensor()]))
        val_dataset = LLVIR(self.annotations_path, train=self.train, transforms=transforms.Compose([transforms.ToTensor()]))

        # Split
        len_total = len(train_dataset)
        len_train = int(len_total * 0.8)
        indices = torch.randperm(len_total).tolist()
        self.train_dataset = torch.utils.data.Subset(train_dataset, indices[:len_train])
        self.val_dataset = torch.utils.data.Subset(val_dataset, indices[len_train:])
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate
        )
    
    def collate(self, batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    
    root_dir = '../data/LLVIP'

    dm = LLVIRDataModule(root_dir, train='train', batch_size=1, num_workers=4)
    dm.setup()

    train_loader = dm.train_dataloader()
    one_batch = next(iter(train_loader))

    images, targets = one_batch

    print(f'Img dim: {images[0].shape}')

    print(f'Max: {images[0].max()}, Min: {images[0].min()}')

    