"""
Part 4: Experiment with different model configurations
Run grid search over backbones, learning rates, and anchor sizes.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

from dataset import PotholeDetectionDataset, collate_fn

def get_model_with_config(backbone='resnet50', anchor_sizes=None):
    """Get model with specified backbone and anchor config"""
    if backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif backbone == 'mobilenet':
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # custom anchors for small objects (potholes are often small)
    if anchor_sizes is not None:
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
        )
        model.rpn.anchor_generator = anchor_generator

    # replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model


def compute_ap_quick(model, loader, device, score_thresh=0.5):
    """Quick AP computation for experiments"""
    model.eval()
    all_dets = []
    all_gt = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            preds = model(images)

            for pred, target in zip(preds, targets):
                keep = pred['scores'] > score_thresh
                boxes = pred['boxes'][keep].cpu()
                scores = pred['scores'][keep].cpu()
                gt_boxes = target['boxes']

                all_dets.append((boxes, scores))
                all_gt.append(gt_boxes)

    # compute matches
    tp = 0
    fp = 0
    n_gt = sum(len(g) for g in all_gt)

    for (boxes, scores), gt in zip(all_dets, all_gt):
        matched = [False] * len(gt)
        for box in boxes:
            best_iou = 0
            best_idx = -1
            for i, g in enumerate(gt):
                iou = compute_iou_tensor(box, g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= 0.5 and best_idx >= 0 and not matched[best_idx]:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / n_gt if n_gt > 0 else 0

    # rough AP estimate
    return precision * recall  # simplified


def compute_iou_tensor(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def train_quick(model, train_loader, val_loader, device, lr=0.005, epochs=5):
    """Quick training for experiments"""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        # validate
        model.train()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # load data
    train_dataset = PotholeDetectionDataset(split='train', augment=True)
    val_dataset = PotholeDetectionDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # experiment configs
    experiments = [
        {'name': 'resnet50_default', 'backbone': 'resnet50', 'lr': 0.005, 'anchors': None},
        {'name': 'resnet50_small_anchors', 'backbone': 'resnet50', 'lr': 0.005,
         'anchors': ((16,), (32,), (64,), (128,), (256,))},
        {'name': 'resnet50_lr_low', 'backbone': 'resnet50', 'lr': 0.001, 'anchors': None},
        {'name': 'mobilenet_default', 'backbone': 'mobilenet', 'lr': 0.005, 'anchors': None},
    ]

    results = []

    print("Running experiments (5 epochs each)...\n")
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: {exp['name']}")
        print(f"  backbone: {exp['backbone']}")
        print(f"  lr: {exp['lr']}")
        print(f"  custom anchors: {exp['anchors'] is not None}")

        model = get_model_with_config(exp['backbone'], exp['anchors'])
        model.to(device)

        val_loss = train_quick(model, train_loader, val_loader, device,
                               lr=exp['lr'], epochs=5)

        # quick AP estimate
        ap = compute_ap_quick(model, val_loader, device)

        result = {
            'name': exp['name'],
            'val_loss': val_loss,
            'ap_estimate': ap
        }
        results.append(result)

        print(f"  -> val_loss: {val_loss:.4f}, AP estimate: {ap:.4f}")

    # save results
    os.makedirs('results/part_4', exist_ok=True)
    with open('results/part_4/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Experiment Summary:")
    print(f"{'='*50}")

    best = min(results, key=lambda x: x['val_loss'])
    for r in sorted(results, key=lambda x: x['val_loss']):
        marker = " <-- best" if r == best else ""
        print(f"  {r['name']}: loss={r['val_loss']:.4f}{marker}")

    print(f"\nBest config: {best['name']}")
    print("Saved results to results/part_4/experiment_results.json")


if __name__ == '__main__':
    main()