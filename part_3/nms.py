import pickle
import numpy as np
from utils import compute_iou

def nms(detections, iou_threshold=0.5):
    """Apply non-maximum suppression to detections"""
    if len(detections) == 0:
        return []
    
    # sort by score (descending)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    keep = []

    while len(detections) > 0:
        # take highest scring box
        best = detections[0]
        keep.append(best)

        # remove boxes with high IoU
        detections = [
            det for det in detections[1:]
            if compute_iou(best['bbox'], det['bbox']) < iou_threshold
        ]

    return keep

def main():
    # load detections
    detections_path = 'results/part_3/detections.pkl'
    output_path = 'results/part_3/detections_nms.pkl'

    print("Loading detections...")
    with open(detections_path, 'rb') as f:
        all_detections = pickle.load(f)

    print(f"Applying NMS...")

    # apply NMS per image
    all_detections_nms = {}
    total_before = 0
    total_after = 0

    for img_name, dets in all_detections.items():
        total_before += len(dets)
        nms_dets = nms(dets, iou_threshold=0.5)
        all_detections_nms[img_name] = nms_dets
        total_after += len(nms_dets)

    # save
    with open(output_path, 'wb') as f:
        pickle.dump(all_detections_nms, f)

    print(f"\nResults:")
    print(f"  Before NMS: {total_before} detections ({total_before/len(all_detections):.1f} per image)")
    print(f"  After NMS: {total_after} detections ({total_after/len(all_detections):.1f} per image)")
    print(f"  Removed: {total_before - total_after} ({100*(total_before-total_after)/total_before:.1f}%)")
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()