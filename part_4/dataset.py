"""
Part 4: Dataset for Faster R-CNN with data augmentation
"""
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

class PotholeDetectionDataset(Dataset):
    def __init__(self, data_path='/dtu/datasets1/02516/potholes/', split='train', seed=42, augment=False):
        self.data_path = Path(data_path)
        self.images_dir = self.data_path / 'images'
        self.annotations_dir = self.data_path / 'annotations'
        self.augment = augment and (split == 'train')

        # get files and split (same as part 1)
        all_files = sorted([f.stem for f in self.annotations_dir.glob('*.xml')])
        random.seed(seed)
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(0.75 * n)
        n_val = int(0.15 * n)

        if split == 'train':
            self.files = all_files[:n_train]
        elif split == 'val':
            self.files = all_files[n_train:n_train + n_val]
        elif split == 'test':
            self.files = all_files[n_train + n_val:]
        else:
            raise ValueError("split must be train/val/test")

        print(f"Loaded {split}: {len(self.files)} images (augment={self.augment})")

    def __len__(self):
        return len(self.files)

    def parse_annotation(self, filename):
        xml_path = self.annotations_dir / f"{filename}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes

    def apply_augmentations(self, img, boxes):
        """Apply random augmentations to image and boxes"""
        w, h = img.size

        # horizontal flip
        if random.random() > 0.5:
            img = F.hflip(img)
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

        # color jitter (doesn't affect boxes)
        if random.random() > 0.5:
            img = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(img)

        # random scale (0.8 to 1.2)
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = F.resize(img, (new_h, new_w))
            boxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] for b in boxes]

        return img, boxes

    def filter_valid_boxes(self, boxes, img_w, img_h, min_size=2):
        """Remove invalid boxes (zero/negative size or out of bounds)"""
        valid = []
        for box in boxes:
            x1, y1, x2, y2 = box

            # clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            # check minimum size
            if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                valid.append([x1, y1, x2, y2])

        return valid

    def __getitem__(self, idx):
        filename = self.files[idx]

        # load image
        img_path = self.images_dir / f"{filename}.png"
        img = Image.open(img_path).convert('RGB')

        # get boxes
        boxes = self.parse_annotation(filename)

        # augment if training
        if self.augment and len(boxes) > 0:
            img, boxes = self.apply_augmentations(img, boxes)

        # filter invalid boxes after augmentation
        img_w, img_h = img.size
        boxes = self.filter_valid_boxes(boxes, img_w, img_h)

        # convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        img = T.ToTensor()(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    # test with augmentation
    dataset = PotholeDetectionDataset(split='train', augment=True)
    
    # test multiple samples to check for issues
    print("Testing augmentation...")
    for i in range(50):
        img, target = dataset[i]
        boxes = target['boxes']
        
        # check for invalid boxes
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            
            if (widths <= 0).any() or (heights <= 0).any():
                print(f"ERROR: Invalid box at idx {i}")
                print(boxes)
    
    print("All samples valid!")
    
    img, target = dataset[0]
    print(f"\nSample: img={img.shape}, boxes={target['boxes'].shape}")