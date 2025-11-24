import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataset import ProposalDataset, BalancedBatchSampler

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (crops, labels) in enumerate(loader):
        crops, labels = crops.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(crops)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print every 50 batches
        if (i + 1) % 50 == 0:
            print(f"Batch {i+1}/{len(loader)}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for crops, labels in loader:
            crops, labels = crops.to(device), labels.to(device)
            outputs = model(crops)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    # paths
    train_proposals = 'results/part_1/train_labeled_proposals.pkl'
    val_proposals = 'results/part_1/val_labeled_proposals.pkl'
    images_dir = '/dtu/datasets1/02516/potholes/images'

    # hyperparameters
    batch_size = 64
    num_epochs = 20
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # load datasets
    print("\nLoading training data...")
    train_dataset = ProposalDataset(train_proposals, images_dir, input_size=64)

    print("\nLoading validation data...")
    val_dataset = ProposalDataset(val_proposals, images_dir, input_size=64)

    # create balanced sampler for training
    train_sampler = BalancedBatchSampler(
        train_dataset.pos_idx,
        train_dataset.neg_idx,
        batch_size=batch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4
    )

    # fixed balanced validation subset (stays same every epoch)
    val_pos_subset = val_dataset.pos_idx # all positives
    val_neg_subset = val_dataset.neg_idx[:len(val_dataset.pos_idx)*3] # 3x negatives
    val_subset_idx = val_pos_subset + val_neg_subset

    val_subset = torch.utils.data.Subset(val_dataset, val_subset_idx)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=True, num_workers=4)

    print(f"Validation subset: {len(val_pos_subset)} pos + {len(val_neg_subset)} neg = {len(val_subset_idx)} total")

    # model, loss, optimizer
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")

    # training loop
    best_val_acc = 0.0
    save_dir = 'results/part_2'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {save_dir}/best_model.pth")


if __name__ == '__main__':
    main()