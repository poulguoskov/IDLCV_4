import os
import pickle
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms

class ProposalDataset(Dataset):
    """Dataset for loading and cropping proposals from part 1"""
    def __init__(self, proposals_path, images_dir, input_size=64):
        self.images_dir = images_dir
        self.input_size = input_size

        # load proposals
        with open(proposals_path, 'rb') as f:
            data = pickle.load(f)

        # flatten to list of sample
        self.samples = []
        self.pos_idx = []
        self.neg_idx = []

        for img_name, img_data in data.items():
            for bbox, label in zip(img_data['proposals'], img_data['labels']):
                idx = len(self.samples)
                self.samples.append({
                    'image': img_name,
                    'bbox': bbox,
                    'label': label
                })

                if label == 1:
                    self.pos_idx.append(idx)
                else:
                    self.neg_idx.append(idx)

        n_pos = len(self.pos_idx)
        n_neg = len(self.neg_idx)
        print(f"Loaded {len(self.samples)} proposals: {n_pos} positive, {n_neg} negative")

        # standard transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load image
        img_name = sample['image']
        if not img_name.endswith('png'):
            img_name = f"{img_name}.png"
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # crop proposals
        x, y, w, h = sample['bbox']
        crop = img.crop((x, y, x+w, y+h))

        # transform
        crop = self.transform(crop)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return crop, label
    
class BalancedBatchSampler(Sampler):
    """Samples batches with 25% positive, 75% negative (as suggested in lecture)"""

    def __init__(self, pos_idx, neg_idx, batch_size=64):
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = batch_size

        # 25% positive per batch
        self.n_pos = batch_size // 4
        self.n_neg = batch_size - self.n_pos

        # number of batches per epoch
        self.n_batches = min(
            len(pos_idx) // self.n_pos,
            len(neg_idx) // self.n_neg
        )

        print(f"Balanced sampler: {self.n_batches} batches, {self.n_pos} pos + {self.n_neg} neg per batch")

    def __iter__(self):
        # shuffle at start of epoch
        pos = self.pos_idx.copy()
        neg = self.neg_idx.copy()
        random.shuffle(pos)
        random.shuffle(neg)

        for i in range(self.n_batches):
            # get samples for this batch
            pos_batch = pos[i*self.n_pos:(i+1)*self.n_pos]
            neg_batch = neg[i*self.n_neg:(i+1)*self.n_neg]

            # combine and shuffle
            batch = pos_batch + neg_batch
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
    
if __name__ == '__main__':
    # quick test
    dataset = ProposalDataset(
        proposals_path='results/part_1/train_labeled_proposals.pkl',
        images_dir='/dtu/datasets1/02516/potholes/images',
        input_size=64
    )

    print(f"\nDataset size: {len(dataset)}")

    # test loading a sample
    crop, label = dataset[0]
    print(f"Sample: crop shape={crop.shape}, label={label}")

    # test balanced sampler
    sampler = BalancedBatchSampler(
        dataset.pos_idx,
        dataset.neg_idx,
        batch_size=64
    )

    # check first batch composition
    batch = next(iter(sampler))
    n_pos = sum(1 for i in batch if i in dataset.pos_idx)
    print(f"\nFirst batch: {len(batch)} samples, {n_pos} positive ({100*n_pos/len(batch):.0f}%)")