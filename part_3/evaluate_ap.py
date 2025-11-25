import pickle
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import random
import matplotlib.pyplot as plt

from utils import compute_iou, get_test_files, parse_annotation

def compute_ap(detections, ground_truths, iou_threshold=0.5):
    """Compute average precision"""
    # sort detections by score
    detections = sorted(detections, key=lambda x: x[2], reverse=True)

    # track which GT boxes have been matched
    gt_matched = {img: [False] * len(boxes) for img, boxes in ground_truths.items()}

    tp = []
    fp = []

    for img_name, det_box, score in detections:
        gt_boxes = ground_truths.get(img_name, [])

        # find best matching GT box
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # check if detections is TP or FP
        if best_iou >= iou_threshold and not gt_matched[img_name][best_gt_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[img_name][best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
        
    # compute cumulative TP and FP
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # total number of GT boxes
    n_gt = sum(len(boxes) for boxes in ground_truths.values())

    # compute precision and recall
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # compute AP (area under PR curve)
    # use 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) > 0 else 0) / 11

    return ap, precisions, recalls

def main():
    data_path = '/dtu/datasets1/02516/potholes/'
    detections_path = 'results/part_3/detections_nms.pkl'

    # load detections
    print("Loading detections...")
    with open(detections_path, 'rb') as f:
        all_detections = pickle.load(f)

    # load ground truth
    print("Loading ground truth...")
    test_files = get_test_files(data_path, seed=42)
    annotations_dir = Path(data_path) / 'annotations'

    ground_truths = {}
    for filename in test_files:
        xml_path = annotations_dir / f"{filename}.xml"
        ground_truths[filename] = parse_annotation(xml_path)

    # flatten detections to list
    detections_list = []
    for img_name, dets in all_detections.items():
        for det in dets:
            detections_list.append((img_name, det['bbox'], det['score']))

    # compute AP
    print("Computing average precision...")
    ap, precisions, recalls = compute_ap(detections_list, ground_truths, iou_threshold=0.5)

    print(f"\n{'='*50}")
    print(f"Average precision @ IoU=0.5: {ap:.4f} ({100*ap:.2f}%)")
    print(f"{'='*50}")
    
    # plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={ap:.4f})')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/part_3/pr_curve.png', dpi=150)
    print(f"\nPrecision-Recall curve saved to results/part_3/pr_curve.png")

    # print some stats
    n_detections = len(detections_list)
    n_gt = sum(len(boxes) for boxes in ground_truths.values())
    print(f"\nStats:")
    print(f"  Total detections: {n_detections}")
    print(f"  Total ground truth boxes: {n_gt}")
    print(f"  Images: {len(test_files)}")


if __name__ == '__main__':
    main()