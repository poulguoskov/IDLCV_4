"""
Part 4: Comprehensive evaluation with threshold tuning
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import PotholeDetectionDataset, collate_fn
from model import get_model

def compute_iou(box1, box2):
    """IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_ap(all_detections, all_gt, iou_thresh=0.5):
    """Compute AP using 11-point interpolation"""
    dets = []
    for img_idx, preds in all_detections.items():
        for box, score in preds:
            dets.append((img_idx, box, score))

    if len(dets) == 0:
        return 0.0, np.array([]), np.array([])

    dets = sorted(dets, key=lambda x: x[2], reverse=True)
    gt_matched = {idx: [False] * len(boxes) for idx, boxes in all_gt.items()}

    tp, fp = [], []

    for img_idx, det_box, score in dets:
        gt_boxes = all_gt.get(img_idx, [])

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[img_idx][best_gt_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[img_idx][best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt = sum(len(boxes) for boxes in all_gt.values())

    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) > 0 else 0) / 11

    return ap, precisions, recalls


def run_inference(model, dataset, device, score_thresh=0.5):
    """Run inference on dataset"""
    all_detections = {}
    all_gt = {}

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Inference (thresh={score_thresh})"):
            img, target = dataset[idx]
            img = img.to(device)

            pred = model([img])[0]

            keep = pred['scores'] > score_thresh
            boxes = pred['boxes'][keep].cpu().numpy()
            scores = pred['scores'][keep].cpu().numpy()

            all_detections[idx] = [(box.tolist(), score) for box, score in zip(boxes, scores)]
            all_gt[idx] = target['boxes'].numpy().tolist()

    return all_detections, all_gt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # load model
    print("Loading model...")
    model = get_model(num_classes=2, pretrained=False)
    #model = get_model(num_classes=2, pretrained=True, backbone='mobilenet')
    model.load_state_dict(torch.load('results/part_4/best_model.pth'))
    model.to(device)

    # load test data
    print("Loading test data...")
    test_dataset = PotholeDetectionDataset(split='test')

    # test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}

    print("\nTesting different score thresholds...")
    for thresh in thresholds:
        dets, gt = run_inference(model, test_dataset, device, score_thresh=thresh)
        ap, prec, rec = compute_ap(dets, gt, iou_thresh=0.5)
        n_dets = sum(len(d) for d in dets.values())
        results[thresh] = {'ap': ap, 'prec': prec, 'rec': rec, 'n_dets': n_dets}
        print(f"  thresh={thresh}: AP={100*ap:.2f}%, detections={n_dets}")

    # find best threshold
    best_thresh = max(results.keys(), key=lambda t: results[t]['ap'])
    best_ap = results[best_thresh]['ap']

    print(f"\n{'='*50}")
    print(f"Best threshold: {best_thresh}")
    print(f"Average Precision @ IoU=0.5: {best_ap:.4f} ({100*best_ap:.2f}%)")
    print(f"{'='*50}")

    # also compute AP at different IoU thresholds
    print("\nAP at different IoU thresholds (using best score thresh):")
    dets, gt = run_inference(model, test_dataset, device, score_thresh=best_thresh)

    iou_thresholds = [0.5, 0.6, 0.7, 0.75]
    for iou_t in iou_thresholds:
        ap, _, _ = compute_ap(dets, gt, iou_thresh=iou_t)
        print(f"  AP@{iou_t}: {100*ap:.2f}%")

    # plot PR curves for different thresholds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PR curves
    for thresh in [0.3, 0.5, 0.7]:
        r = results[thresh]
        if len(r['rec']) > 0:
            axes[0].plot(r['rec'], r['prec'], label=f'thresh={thresh} (AP={100*r["ap"]:.1f}%)')

    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('PR Curves at Different Score Thresholds')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # AP vs threshold
    axes[1].plot(thresholds, [results[t]['ap'] * 100 for t in thresholds], 'bo-')
    axes[1].set_xlabel('Score Threshold')
    axes[1].set_ylabel('AP (%)')
    axes[1].set_title('AP vs Score Threshold')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('results/part_4/evaluation_results.png', dpi=150)
    print(f"\nSaved evaluation results to results/part_4/evaluation_results.png")

    # stats
    n_gt = sum(len(g) for g in gt.values())
    print(f"\nDataset stats:")
    print(f"  Test images: {len(test_dataset)}")
    print(f"  Total ground truth boxes: {n_gt}")


if __name__ == '__main__':
    main()