import os
import pickle
import random
from pathlib import Path
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append('part_1')
sys.path.append('part_2')

from extract_proposals import extract_proposals
from model import SimpleCNN

def get_test_files(data_path='/dtu/datasets1/02516/potholes/', seed=42):
    """Get test files using same split as Part 1"""
    annotations_dir = Path(data_path) / 'annotations'
    
    # get all files and shuffle (same as data_loader.py)
    all_files = sorted([f.stem for f in annotations_dir.glob('*.xml')])
    random.seed(seed)
    random.shuffle(all_files)
    
    # 75/15/15 split (same as data_loader.py)
    n = len(all_files)
    n_train = int(0.75 * n)
    n_val = int(0.15 * n)
    
    test_files = all_files[n_train + n_val:]
    return test_files

def detect_potholes(image_path, model, device, transform, threshold=0.5):
    """Run full detection pipeline on a single image"""
    # load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # extract proposals
    proposals = extract_proposals(img_rgb, scale=50, sigma=0.9, min_size=50)

    # classify each proposals
    detections = []
    model.eval()

    with torch.no_grad():
        for bbox in proposals:
            x, y, w, h = bbox

            # crop proposal
            crop = Image.fromarray(img_rgb[y:y+h, x:x+w])
            crop_tensor = transform(crop).unsqueeze(0).to(device)

            # get prediction
            output = model(crop_tensor)
            probs = torch.softmax(output, dim=1)[0]

            # if predicted as pothole (class 1)
            if probs[1] > threshold:
                detections.append({
                    'bbox': bbox,
                    'score': probs[1].item()
                })

    return detections

def main():
    # paths
    data_path = '/dtu/datasets1/02516/potholes/'
    images_dir = os.path.join(data_path, 'images')
    model_path = 'results/part_2/best_model.pth'
    output_path = 'results/part_3/detections.pkl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # load model
    print("Loading model...")
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}\n")
    
    # setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # get test files (same split as part 1)
    test_files = get_test_files(data_path, seed=42)
    print(f"Processing {len(test_files)} test images...")

    # run detection on each image
    all_detections = {}

    for i, filename in enumerate(test_files):
        img_path = os.path.join(images_dir, f"{filename}.png")

        detections = detect_potholes(img_path, model, device, transform, threshold=0.5)
        all_detections[filename] = detections

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_files)} images")

    # save detections
    os.makedirs('results/part_3', exist_ok=True)
    with open(output_path,'wb') as f:
        pickle.dump(all_detections, f)

    print(f"\nDetections saved to {output_path}")

    # print stats
    total_dets = sum(len(dets) for dets in all_detections.values())
    avg_dets = total_dets / len(test_files)
    print(f"Total detections: {total_dets}")
    print(f"Average per image: {avg_dets:.2f}")

if __name__ == '__main__':
    main()