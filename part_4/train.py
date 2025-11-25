"""
Part 4: Train Faster R-CNN with augmentation and LR warmup
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import PotholeDetectionDataset, collate_fn
from model import get_model

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Linear warmup scheduler"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, loader, optimizer, device, epoch, warmup_scheduler=None):
    model.train()
    total_loss = 0
    loss_cls = 0
    loss_box = 0
    loss_obj = 0
    loss_rpn = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_loss += losses.item()
        loss_cls += loss_dict.get('loss_classifier', torch.tensor(0)).item()
        loss_box += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
        loss_obj += loss_dict.get('loss_objectness', torch.tensor(0)).item()
        loss_rpn += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()

        pbar.set_postfix({'loss': f'{losses.item():.3f}'})

    n = len(loader)
    return {
        'total': total_loss / n,
        'cls': loss_cls / n,
        'box': loss_box / n,
        'obj': loss_obj / n,
        'rpn': loss_rpn / n
    }


def validate(model, loader, device):
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(loader)


def plot_losses(train_losses, val_losses, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # total loss
    axes[0].plot(train_losses['total'], label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)

    # component losses
    axes[1].plot(train_losses['cls'], label='Classifier')
    axes[1].plot(train_losses['box'], label='Box Reg')
    axes[1].plot(train_losses['obj'], label='Objectness')
    axes[1].plot(train_losses['rpn'], label='RPN Box')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    # hyperparams
    batch_size = 4
    num_epochs = 15
    lr = 0.005
    warmup_epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # datasets with augmentation
    print("\nLoading datasets...")
    train_dataset = PotholeDetectionDataset(split='train', augment=True)
    val_dataset = PotholeDetectionDataset(split='val', augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # model - use small anchors based on experiments
    print("\nLoading pretrained Faster R-CNN (small anchors)...")
    model = get_model(num_classes=2, pretrained=True)
    #model = get_model(num_classes=2, pretrained=True, backbone='mobilenet')
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # lr scheduler - reduce on plateau
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # warmup scheduler for first epoch
    warmup_iters = warmup_epochs * len(train_loader)
    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.1)

    # training
    os.makedirs('results/part_4', exist_ok=True)
    best_val_loss = float('inf')

    train_history = {'total': [], 'cls': [], 'box': [], 'obj': [], 'rpn': []}
    val_history = []

    print(f"\nTraining for {num_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {lr}")
    print(f"  Warmup epochs: {warmup_epochs}")

    for epoch in range(1, num_epochs + 1):
        # use warmup scheduler for first epoch
        ws = warmup_scheduler if epoch == 1 else None

        losses = train_one_epoch(model, train_loader, optimizer, device, epoch, ws)
        val_loss = validate(model, val_loader, device)

        # update main scheduler after warmup
        if epoch > warmup_epochs:
            main_scheduler.step(val_loss)

        # log
        for k in train_history:
            train_history[k].append(losses[k])
        val_history.append(val_loss)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train={losses['total']:.4f}, val={val_loss:.4f}, lr={cur_lr:.6f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/part_4/best_model.pth')
            print(f"  -> Saved best model")

        # save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }, f'results/part_4/checkpoint_epoch{epoch}.pth')

    # plot losses
    plot_losses(train_history, val_history, 'results/part_4/training_curves.png')
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print("Saved training curves to results/part_4/training_curves.png")


if __name__ == '__main__':
    main()