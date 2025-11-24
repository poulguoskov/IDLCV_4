import pickle
import numpy as np

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = max(box1_x2, box2_x2)
    yi2 = max(box1_y2, box2_y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

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