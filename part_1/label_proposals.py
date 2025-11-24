"""
Part 1: Task 4 - Label proposals for training

Assign labels based on IoU with ground truth:
- Positive (1): IoU >= 0.5 (overlaps with pothole)
- Negative (0): IoU < 0.5 (background)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from data_loader import PotholesDataset
from part_1.eval_proposals import compute_iou

def label_proposals(proposals_dict, dataset, iou_thresh=0.5):
    """Label each proposals as positive or negative based on IoU"""
    labeled_data = {}
    n_pos = 0
    n_neg = 0

    for idx in tqdm(range(len(dataset))):
        _, ann = dataset[idx]
        filename = ann['filename']
        gt_boxes = ann['boxes']

        if filename not in proposals_dict:
            continue

        props = proposals_dict[filename]['proposals']
        labels = []

        for prop in props:
            # find max IoU with any GT box
            max_iou = 0.0
            for gt in gt_boxes:
                iou = compute_iou(prop, gt)
                if iou > max_iou:
                    max_iou = iou

            # assign label
            if max_iou >= iou_thresh:
                labels.append(1)
                n_pos += 1
            else:
                labels.append(0)
                n_neg += 1

        labeled_data[filename] = {
            'proposals': props,
            'labels': np.array(labels),
            'gt_boxes': gt_boxes
        }

    return labeled_data, n_pos, n_neg

def plot_examples(labeled_data, dataset, n=4):
    """Show some examples of labeled proposals."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    count = 0
    for i in range(len(dataset)):
        if count >= n:
            break
        
        img, ann = dataset[i]
        filename = ann['filename']
        
        if filename not in labeled_data:
            continue
        
        data = labeled_data[filename]
        if np.sum(data['labels']) == 0:  # skip images with no positives
            continue
        
        ax = axes[count]
        ax.imshow(img)
        
        # draw GT boxes (green)
        for box in data['gt_boxes']:
            rect = patches.Rectangle(
                (box['xmin'], box['ymin']),
                box['xmax'] - box['xmin'],
                box['ymax'] - box['ymin'],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
        
        # draw positive proposals (blue)
        pos_idx = np.where(data['labels'] == 1)[0][:5]
        for idx in pos_idx:
            x, y, w, h = data['proposals'][idx]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        
        # draw some negative proposals (red)
        neg_idx = np.where(data['labels'] == 0)[0][:3]
        for idx in neg_idx:
            x, y, w, h = data['proposals'][idx]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1, edgecolor='red', facecolor='none', linestyle=':'
            )
            ax.add_patch(rect)
        
        n_pos = np.sum(data['labels'])
        n_neg = len(data['labels']) - n_pos
        ax.set_title(f"{n_pos} positive, {n_neg} negative")
        ax.axis('off')
        
        count += 1
    
    plt.suptitle('Green=GT, Blue=Positive, Red=Negative', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/part_1/figures/labeled_examples.png', dpi=150)
    print("Saved examples to results/part_1/figures/labeled_examples.png")

if __name__ == '__main__':
    os.makedirs('results/part_1', exist_ok=True)

    print("\nLabeling proposals for training...")
    print("Using IoU >= 0.5 for positive labels\n")

    # label all splits
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} set...")

        dataset = PotholesDataset(split=split)

        # load optimized proposals
        with open(f'results/part_1/optimized_{split}_proposals.pkl', 'rb') as f:
            proposals_dict = pickle.load(f)

        # label them
        labeled_data, n_pos, n_neg = label_proposals(proposals_dict, dataset)

        # save
        with open(f'results/part_1/{split}_labeled_proposals.pkl', 'wb') as f:
            pickle.dump(labeled_data, f)

        total = n_pos + n_neg
        print(f"  {split}: {n_pos:,} positive ({n_pos/total*100:.1f}%), {n_neg:,} negative")
        print(f"  Ratio: 1:{n_neg/n_pos:.1f}\n")
        
        # visualize train set
        if split == 'train':
            plot_examples(labeled_data, dataset)
    
    print("Done! Labeled proposals saved.")