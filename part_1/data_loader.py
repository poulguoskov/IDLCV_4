"""
Part 1: Task 1 - Data Familiarization

Data loader and visualization for potholes dataset.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PotholesDataset:
    def __init__(self, data_path='/dtu/datasets1/02516/potholes/', split='train', seed=42):
        self.data_path = Path(data_path)
        self.images_dir = self.data_path / 'images'
        self.annotations_dir = self.data_path / 'annotations'
        self.split = split

        # get all files and shuffle
        all_files = sorted([f.stem for f in self.annotations_dir.glob('*.xml')])
        random.seed(seed)
        random.shuffle(all_files)

        # 75/15/15 split
        n = len(all_files)
        n_train = int(0.75 * n)
        n_val = int(0.15 * n)

        self.train_files = all_files[:n_train]
        self.val_files = all_files[n_train:n_train + n_val]
        self.test_files = all_files[n_train + n_val:]

        if split == 'train':
            self.files = self.train_files
        elif split == 'val':
            self.files = self.val_files
        elif split == 'test':
            self.files = self.test_files
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        print(f"Loaded {split}: {len(self.files)} images")

    def __len__(self):
        return len(self.files)
    
    def parse_annotation(self, filename):
        xml_path = self.annotations_dir / f"{filename}.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            boxes.append({
                'xmin': int(bbox.find('xmin').text),
                'ymin': int(bbox.find('ymin').text),
                'xmax': int(bbox.find('xmax').text),
                'ymax': int(bbox.find('ymax').text),
            })

        return {'filename': filename, 'width': width, 'height': height, 'boxes': boxes}
    
    def load_image(self, filename):
        return Image.open(self.images_dir / f"{filename}.png").convert("RGB")
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        return self.load_image(filename), self.parse_annotation(filename)
    
    def get_all_annotations(self):
        return [self.parse_annotation(f) for f in self.files]
    
def visualize_sample(image, annotation, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.imshow(image)
    for box in annotation['boxes']:
        rect = patches.Rectangle(
            (box['xmin'], box['ymin']),
            box['xmax'] - box['xmin'],
            box['ymax'] - box['ymin'],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title(f"{annotation['filename']} - {len(annotation['boxes'])} potholes")
    plt.axis('off')
    return ax

def visualize_grid(dataset, n=6, seed=42):
    random.seed(seed)
    indices = np.random.choice(len(dataset), n, replace=False)
    
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, annotation = dataset[idx]
        visualize_sample(image, annotation, ax=axes[i])

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

def plot_bbox_stats(dataset):
    annotations = dataset.get_all_annotations()
    
    n_bboxes = [len(ann['boxes']) for ann in annotations]
    widths, heights = [], []

    for ann in annotations:
        for box in ann['boxes']:
            widths.append(box['xmax'] - box['xmin'])
            heights.append(box['ymax'] - box['ymin'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(n_bboxes, bins=20, edgecolor='black')
    axes[0].set_xlabel('Potholes per image')
    axes[0].set_title(f'Mean: {np.mean(n_bboxes):.1f}')

    axes[1].hist(widths, bins=30, edgecolor='black')
    axes[1].set_xlabel('Width (px)')
    axes[1].set_title(f'Mean: {np.mean(widths):.1f}')
    
    axes[2].hist(heights, bins=30, edgecolor='black')
    axes[2].set_xlabel('Height (px)')
    axes[2].set_title(f'Mean: {np.mean(heights):.1f}')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    os.makedirs('results/part_1/figures', exist_ok=True)

    dataset = PotholesDataset(split='train')
    img, ann = dataset[0]
    print(f"Image size: {img.size}, Potholes: {len(ann['boxes'])}")

    fig = visualize_grid(dataset)
    fig.savefig('results/part_1/figures/samples.png', dpi=150)

    fig = plot_bbox_stats(dataset)
    fig.savefig('results/part_1/figures/stats.png', dpi=150)
    print("Saved figures")