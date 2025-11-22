"""
Part 1: Task 2 - Extract object proposals using selective search
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import selectivesearch
from data.data_loader import PotholesDataset

def extract_proposals(image, scale=500, sigma=0.9, min_size=20):
    """Run selective search on an image"""
    img_array = np.array(image, dtype=np.uint8)
    _, regions = selectivesearch.selective_search(img_array, scale=scale, sigma=sigma, min_size=min_size)

    proposals = []
    for r in regions:
        x, y, w, h = r['rect']
        if w > 0 and h > 0:
            proposals.append([x, y, w, h])
    return np.array(proposals)

if __name__ == "__main__":
    os.makedirs('results/part_1', exist_ok=True)

    SCALE = 500
    SIGMA = 0.9
    MIN_SIZE = 20

    print(f"Params: scale={SCALE}, sigma={SIGMA}, min_size={MIN_SIZE}\n")

    for split in ['train', 'val', 'test']:
        dataset = PotholesDataset(split=split)

        all_proposals = {}
        n_proposals_list = []

        for idx in tqdm(range(len(dataset)), desc=f"Extracting {split} set"):
            img, ann = dataset[idx]
            proposals = extract_proposals(img, scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE)
            n_proposals_list.append(len(proposals))

            all_proposals[ann['filename']] = {
                'proposals': proposals,
                'width': ann['width'],
                'height': ann['height']
            }

        #save
        with open(f'results/part_1/{split}_proposals.pkl', 'wb') as f:
            pickle.dump(all_proposals, f)

        print(f"{split} set: {len(dataset)} images, mean {np.mean(n_proposals_list):.0f} proposals/image\n")