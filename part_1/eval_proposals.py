"""
Part 1: Task 3 - Evaluate proposals quality using recall and MABO
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from data_loader import PotholesDataset

def compute_iou(proposal, gt_box):
    """Compute IoU between proposal and gt_box dict."""
    px, py, pw, ph = proposal

    x1 = max(px, gt_box['xmin'])
    y1 = max(py, gt_box['ymin'])
    x2 = min(px + pw, gt_box['xmax'])
    y2 = min(py + ph, gt_box['ymax'])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    prop_area = pw * ph
    gt_area = (gt_box['xmax'] - gt_box['xmin']) * (gt_box['ymax'] - gt_box['ymin'])
    union = prop_area + gt_area - inter

    return inter / union if union > 0 else 0

def compute_recall(proposals, gt_boxes, iou_thresh=0.5):
    """Fraction of gt boxes matched by at least one proposal."""
    if len(gt_boxes) == 0:
        return 1.0
    if len(proposals) == 0:
        return 0.0
    
    matched = 0
    for gt in gt_boxes:
        for prop in proposals:
            if compute_iou(prop, gt) >= iou_thresh:
                matched += 1
                break
    return matched / len(gt_boxes)

def compute_mabo(proposals, gt_boxes):
    """Mean avg best overlap - avg of best IoU for each gt box."""
    if len(gt_boxes) == 0:
        return 1.0
    if len(proposals) == 0:
        return 0.0
    
    best_ious = []
    for gt in gt_boxes:
        ious = [compute_iou(prop, gt) for prop in proposals]
        best_ious.append(max(ious))

    return np.mean(best_ious)

def visualize_proposals_with_gt(image, proposals, gt_boxes, max_proposals=50, save_path=None):
    """Visualize proposals (blue) and ground truth boxes (red) on image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    
    # Draw proposals (blue, transparent)
    n_show = min(max_proposals, len(proposals))
    for i in range(n_show):
        x, y, w, h = proposals[i]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1, edgecolor='green', facecolor='none', alpha=0.5
        )
        ax.add_patch(rect)
    
    # Draw ground truth (red, solid)
    for box in gt_boxes:
        rect = patches.Rectangle(
            (box['xmin'], box['ymin']),
            box['xmax'] - box['xmin'],
            box['ymax'] - box['ymin'],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_title(f"Blue: {n_show}/{len(proposals)} proposals | Red: {len(gt_boxes)} GT boxes")
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    dataset = PotholesDataset(split="train")

    with open('results/part_1/train_proposals.pkl', 'rb') as f:
        proposals_dict = pickle.load(f)

    print("Evaluating proposals on train set...\n")

    recalls = []
    mabos = []
    n_proposals_list = []

    for idx in tqdm(range(len(dataset))):
        _, ann = dataset[idx]

        props_data = proposals_dict[ann['filename']]
        proposals = props_data['proposals']
        n_proposals_list.append(len(proposals))

        recall = compute_recall(proposals, ann['boxes'])
        recalls.append(recall)

        mabo = compute_mabo(proposals, ann['boxes'])
        mabos.append(mabo)

    # Results
    print(f"\nResults:")
    print(f"Mean proposals/image: {np.mean(n_proposals_list):.0f}")
    print(f"Mean recall @IoU=0.5: {np.mean(recalls):.3f}")
    print(f"Mean MABO: {np.mean(mabos):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(n_proposals_list, bins=30, edgecolor='black')
    axes[0].axvline(np.mean(n_proposals_list), color='red', linestyle='--')
    axes[0].set_xlabel('Proposals per image')
    axes[0].set_title('Proposal Distribution')
    
    axes[1].hist(recalls, bins=20, edgecolor='black')
    axes[1].axvline(np.mean(recalls), color='red', linestyle='--')
    axes[1].set_xlabel('Recall')
    axes[1].set_title(f'Recall (mean: {np.mean(recalls):.3f})')
    
    axes[2].hist(mabos, bins=20, edgecolor='black')
    axes[2].axvline(np.mean(mabos), color='red', linestyle='--')
    axes[2].set_xlabel('MABO')
    axes[2].set_title(f'MABO (mean: {np.mean(mabos):.3f})')
    
    plt.tight_layout()
    plt.savefig('results/part_1/figures/evaluation.png', dpi=150)
    print("\nSaved results/part_1/figures/evaluation.png")

    # Visualize a few examples
    print("\nGenerating example visualizations...")
    
    for i in [0, 10, 20]:  # visualize 3 images
        img, ann = dataset[i]
        props_data = proposals_dict[ann['filename']]
        
        visualize_proposals_with_gt(
            img, 
            props_data['proposals'],
            ann['boxes'],
            max_proposals=100,
            save_path=f'results/part_1/figures/proposals_example_{i}.png'
        )
    
    print(f"Saved example visualizations to results/part_1/figures/")