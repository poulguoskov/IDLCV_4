"""
Part 4: Visualize Faster R-CNN detections with detailed analysis
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from dataset import PotholeDetectionDataset
from model import get_model

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def visualize_detection(img, pred_boxes, pred_scores, gt_boxes, title='', save_path=None):
    """Detailed visualization with TP/FP/FN analysis"""
    fig, ax = plt.subplots(figsize=(12, 9))

    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    # match predictions to GT
    gt_matched = [False] * len(gt_boxes)
    pred_status = []  # 'tp' or 'fp' for each pred

    for box, score in zip(pred_boxes, pred_scores):
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= 0.5 and best_idx >= 0 and not gt_matched[best_idx]:
            pred_status.append('tp')
            gt_matched[best_idx] = True
        else:
            pred_status.append('fp')

    # draw GT boxes
    for i, box in enumerate(gt_boxes):
        color = 'green' if gt_matched[i] else 'orange'  # orange = missed (FN)
        linestyle = '-' if gt_matched[i] else '--'
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle
        )
        ax.add_patch(rect)

    # draw predictions
    for box, score, status in zip(pred_boxes, pred_scores, pred_status):
        color = 'blue' if status == 'tp' else 'red'  # red = false positive
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 5, f'{score:.2f}', color=color, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # stats
    n_tp = sum(1 for s in pred_status if s == 'tp')
    n_fp = sum(1 for s in pred_status if s == 'fp')
    n_fn = sum(1 for m in gt_matched if not m)

    ax.set_title(f'{title}\nGreen=GT matched, Orange=GT missed (FN), Blue=TP, Red=FP\n'
                 f'TP={n_tp}, FP={n_fp}, FN={n_fn}')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # load model
    print("Loading model...")
    model = get_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load('results/part_4/best_model.pth'))
    model.to(device)
    model.eval()

    # test data
    test_dataset = PotholeDetectionDataset(split='test')

    os.makedirs('results/part_4/figures', exist_ok=True)

    # collect stats
    all_tp, all_fp, all_fn = 0, 0, 0

    print("Generating visualizations...")
    
    # visualize variety of images
    indices = list(range(0, min(20, len(test_dataset)), 2))  # every other image

    for idx in indices:
        img, target = test_dataset[idx]

        with torch.no_grad():
            pred = model([img.to(device)])[0]

        # filter by confidence
        keep = pred['scores'] > 0.5
        pred_boxes = pred['boxes'][keep].cpu().numpy()
        pred_scores = pred['scores'][keep].cpu().numpy()
        gt_boxes = target['boxes'].numpy()

        save_path = f'results/part_4/figures/detection_{idx:03d}.png'
        visualize_detection(img, pred_boxes, pred_scores, gt_boxes,
                           title=f'Image {idx}', save_path=save_path)
        print(f"  Saved {save_path}")

    # create grid summary
    print("\nCreating summary grid...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate([0, 5, 10, 15, 20, 25]):
        if idx >= len(test_dataset):
            axes[i].axis('off')
            continue

        img, target = test_dataset[idx]

        with torch.no_grad():
            pred = model([img.to(device)])[0]

        keep = pred['scores'] > 0.5
        pred_boxes = pred['boxes'][keep].cpu().numpy()
        pred_scores = pred['scores'][keep].cpu().numpy()
        gt_boxes = target['boxes'].numpy()

        axes[i].imshow(img.permute(1, 2, 0).numpy())

        # draw boxes
        for box in gt_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor='green', facecolor='none')
            axes[i].add_patch(rect)

        for box, score in zip(pred_boxes, pred_scores):
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            axes[i].add_patch(rect)

        axes[i].set_title(f'Image {idx}: {len(pred_boxes)} pred, {len(gt_boxes)} GT')
        axes[i].axis('off')

    plt.suptitle('Faster R-CNN Detections (Green=GT, Blue=Pred)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/part_4/figures/detection_grid.png', dpi=150)
    print("Saved results/part_4/figures/detection_grid.png")


if __name__ == '__main__':
    main()