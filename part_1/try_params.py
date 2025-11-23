"""
Quick test of different parameters to see if we can improve recall and mabo.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import PotholesDataset
from part_1.extract_proposals import extract_proposals
from part_1.eval_proposals import compute_iou, compute_recall, compute_mabo
import numpy as np
from tqdm import tqdm

dataset = PotholesDataset(split='train')

# try a few different parameters
configs = [
    {'scale': 500, 'sigma': 0.9, 'min_size': 20},  # baseline
    {'scale': 300, 'sigma': 0.9, 'min_size': 30},
    {'scale': 100, 'sigma': 0.9, 'min_size': 40},
]

print("Testing different parameters...\n")

for config in configs:
    print(f"Testing: scale={config['scale']}, sigma={config['sigma']}, min_size={config['min_size']}")
    
    recalls = []
    mabos = []
    n_props = []

    # test on first 100 images for speed
    for i in tqdm(range(100)):
        img, ann = dataset[i]
        props = extract_proposals(img, **config)
        n_props.append(len(props))

        recall = compute_recall(props, ann['boxes'])
        recalls.append(recall)

        mabo = compute_mabo(props, ann['boxes'])
        mabos.append(mabo)

    print(f"  Recall: {np.mean(recalls):.3f}")
    print(f"  MABO: {np.mean(mabos):.3f}")
    print(f"  Proposals: {np.mean(n_props):.0f}\n")