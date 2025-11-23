# Project 4: Pothole Detection System
02516 Introduction to Deep Learning in Computer Vision - DTU

Three part project to build a pothole detection system using object proposals and a CNN classifier.

## Setup
```bash
conda env create -f environment.yml
conda activate IDLCV
```

Dataset: `/dtu/datasets1/02516/potholes/`

---

## Part 1: Object Proposals

### Task 1: Data Familiarization
Loaded the dataset and visualized samples with ground truth bounding boxes. Split into 498 train, 99 val, 68 test.

<img src="results/part_1/figures/samples.png" alt="Sample Pothole Images" width="500">

### Task 2: Proposal Extraction
Used selective search with scale=500, sigma=0.9, min_size=20. Extracted proposals for all splits (~300 proposals per image on avg).

Results saved as pickle files (Python's serialization format for saving objects):
- `results/part_1/train_proposals.pkl` - 498 images
- `results/part_1/val_proposals.pkl` - 99 images  
- `results/part_1/test_proposals.pkl` - 68 images

Each pickle file contains a dict mapping filename → proposals + image dimensions.

### Task 3: Evaluation
Evaluated proposals using recall and MABO metrics on train set.

**Baseline results (scale=500, sigma=0.9, min_size=20)**
- Mean recall @IoU=0.5: **60.1%**
- Mean MABO: **54.5%**
- Proposals per image: ~306

<img src="results/part_1/figures/proposals_example_10.png" alt="Proposal visualization" width="500">

*Green: proposals, Red: ground truth boxes*

---

**Optional: Parameter optimization**

Baseline recall&mabo seemed low, so tested if different parameters could improve it.

*Manual testing (`try_params.py`):*
- scale=500, min_size=20: 59.8% recall, 54.4% MABO, 302 proposals
- scale=300, min_size=30: 64.3% recall, 57.7% MABO, 312 proposals  
- scale=100, min_size=40: **79.2% recall**, 63.5% MABO, 614 proposals

Smaller scale and larger min_size improves recall&mabo. Decided to run a systematic grid search.

*Systematic optimization (`optimize_params.py`):*
- Grid search: 35 combinations (scale: 50-500, min_size: 10-50, sigma fixed at 0.9)
- Best config: **scale=50, sigma=0.9, min_size=50**
- Results: **81.0% recall**, **63.5% MABO**, ~883 proposals/image

<img src="results/part_1/figures/optimization_plots.png" alt="Optimization results" width="500">

**Big improvement over baseline!** Re-extracted all splits with optimized parameters. Will use `optimized_*_proposals.pkl` for Part 2.

### Task 4: Labeling
TODO

---

## Part 2: CNN Classifier

### Task 1: Build CNN
TODO

### Task 2: Dataloader with Balanced Sampling
TODO

### Task 3: Training
TODO

### Task 4: Evaluation
TODO

---

## Part 3: Testing & Evaluation

### Task 1: Apply CNN to Test Images
TODO

### Task 2: NMS
TODO

### Task 3: Average Precision
TODO