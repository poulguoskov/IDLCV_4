"""
Systematic parameter optimization - grid search over scale and min_size.
Manual testing showed scale and min_size mattered more than sigma, so sigma is fixed here.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import PotholesDataset
from part_1.extract_proposals import extract_proposals
from part_1.eval_proposals import compute_recall, compute_mabo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def run_optimization():
    """Run grid search and visualize results."""
    dataset = PotholesDataset(split='train')

    # grid search
    scales = [500, 300, 200, 150, 100, 75, 50]
    min_sizes = [10, 20, 30, 40, 50]
    sigma = 0.9

    print(f"Grid search: {len(scales)*len(min_sizes)} combinations")
    print(f"Fixed sigma={sigma}\n")

    results = []

    for scale in scales:
        for min_size in min_sizes:
            recalls = []
            mabos = []
            n_props = []

            # evaluate on 100 images for speed
            desc = f"scale={scale:3d}, min_size={min_size:2d}"
            for i in tqdm(range(100), desc=desc, leave=False):
                img, ann = dataset[i]
                props = extract_proposals(img, scale=scale, sigma=sigma, min_size=min_size)

                n_props.append(len(props))
                recalls.append(compute_recall(props, ann['boxes']))
                mabos.append(compute_mabo(props, ann['boxes']))

            result = {
                'scale': scale,
                'min_size': min_size,
                'recall': np.mean(recalls),
                'mabo': np.mean(mabos),
                'n_props': np.mean(n_props)
            }
            results.append(result)
            print(f"{desc} → recall={result['recall']:.3f}, mabo={result['mabo']:.3f}, props={result['n_props']:.0f}")
    
    return results, sigma

def plot_results(results):
    """Visualize optimization results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # prepare data
    scales = sorted(list(set(r['scale'] for r in results)))
    min_sizes = sorted(list(set(r['min_size'] for r in results)))
    
    # 1. Recall heatmap
    recall_grid = np.zeros((len(min_sizes), len(scales)))
    for r in results:
        i = min_sizes.index(r['min_size'])
        j = scales.index(r['scale'])
        recall_grid[i, j] = r['recall']
    
    im = axes[0].imshow(recall_grid, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.8)
    axes[0].set_xticks(range(len(scales)))
    axes[0].set_xticklabels(scales)
    axes[0].set_yticks(range(len(min_sizes)))
    axes[0].set_yticklabels(min_sizes)
    axes[0].set_xlabel('Scale')
    axes[0].set_ylabel('Min Size')
    axes[0].set_title('Recall')
    plt.colorbar(im, ax=axes[0])
    
    # 2. Recall vs proposals scatter
    recalls = [r['recall'] for r in results]
    n_props = [r['n_props'] for r in results]
    colors = [r['scale'] for r in results]
    
    scatter = axes[1].scatter(n_props, recalls, c=colors, s=100, alpha=0.6, cmap='viridis')
    axes[1].set_xlabel('Proposals per image')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall vs Proposals')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Scale')
    
    # 3. Recall vs MABO
    mabos = [r['mabo'] for r in results]
    scatter2 = axes[2].scatter(mabos, recalls, c=colors, s=100, alpha=0.6, cmap='viridis')
    axes[2].set_xlabel('MABO')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Recall vs MABO')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[2], label='Scale')
    
    plt.tight_layout()
    plt.savefig('results/part_1/figures/optimization_plots.png', dpi=150)
    print("\nSaved plots to results/part_1/figures/optimization_plots.png")

def find_best_config(results):
    """Find best configuration. We optimize for recall since we need to capture all potholes."""
    # Note: Could also optimize for MABO (quality) or a weighted combination,
    # but recall is more critical for detection - we need to find the object first
    best = max(results, key=lambda x: x['recall'])
    return best


def reextract_with_optimal(best_config, sigma):
    """Re-extract proposals for all splits with optimal parameters."""
    print("\n" + "="*60)
    print("Re-extracting with optimal parameters")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        dataset = PotholesDataset(split=split)
        
        all_proposals = {}
        n_proposals_list = []
        
        for idx in tqdm(range(len(dataset))):
            img, ann = dataset[idx]
            proposals = extract_proposals(
                img, 
                scale=best_config['scale'],
                sigma=sigma,
                min_size=best_config['min_size']
            )
            n_proposals_list.append(len(proposals))
            
            all_proposals[ann['filename']] = {
                'proposals': proposals,
                'width': ann['width'],
                'height': ann['height']
            }
        
        # save
        save_path = f'results/part_1/optimized_{split}_proposals.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(all_proposals, f)
        
        print(f"Saved {len(all_proposals)} images, mean {np.mean(n_proposals_list):.0f} proposals/img")


if __name__ == '__main__':
    os.makedirs('results/part_1', exist_ok=True)
    
    # run optimization
    results, sigma = run_optimization()
    
    # visualize
    plot_results(results)
    
    # find best
    best = find_best_config(results)
    print("\nBest configuration (optimized for recall):\n")
    print(f"  scale = {best['scale']}")
    print(f"  sigma = {sigma}")
    print(f"  min_size = {best['min_size']}")
    print(f"\nPerformance:")
    print(f"  Recall: {best['recall']:.3f}")
    print(f"  MABO: {best['mabo']:.3f}")
    print(f"  Proposals: {best['n_props']:.0f}")
    
    # re-extract with optimal parameters
    reextract_with_optimal(best, sigma)

    print("Optimization complete!")