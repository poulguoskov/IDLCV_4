import torch
import torch.nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import SimpleCNN
from dataset import ProposalDataset

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for crops, labels in loader:
            crops = crops.to(device)
            outputs = model(crops)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels

def main():
    # paths
    test_proposals = 'results/part_1/test_labeled_proposals.pkl'
    images_dir = '/dtu/datasets1/02516/potholes/images'
    model_path = 'results/part_2/best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # load test data
    print("Loading test data...")
    test_datset = ProposalDataset(test_proposals, images_dir, input_size=64)

    # use all data (no balanced sampling)
    test_loader = DataLoader(
        test_datset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # load model
    print("Loading model...")
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}\n")

    # evaluate
    print("Evaluating on test set...")
    preds, labels = evaluate_model(model, test_loader, device)

    # calculate metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)

    print("="*50)
    print("Test set results")
    print("="*50)
    print(f"Overall accuracy: {100*acc:.2f}%\n")

    print("Per-class metrics:")
    print(f"  Background (class 0):")
    print(f"    Precision: {100*precision[0]:.2f}%")
    print(f"    Recall: {100*recall[0]:.2f}%")
    print(f"    F1 Score: {100*f1[0]:.2f}%")
    
    print(f"\n  Pothole (class 1):")
    print(f"    Precision: {100*precision[1]:.2f}%")
    print(f"    Recall: {100*recall[1]:.2f}%")
    print(f"    F1 Score: {100*f1[1]:.2f}%")

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion matrix:")
    print(cm)

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Pothole'],
                yticklabels=['Background', 'Pothole'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('results/part_2/confusion_matrix.png', dpi=150)
    print(f"\nConfusion matrix saved to results/part_2/confusion_matrix.png")


if __name__ == '__main__':
    main()