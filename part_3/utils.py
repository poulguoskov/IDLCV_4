import random
from pathlib import Path
import xml.etree.ElementTree as ET

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
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def get_test_files(data_path='/dtu/datasets1/02516/potholes/', seed=42):
    """Get test files using same split as Part 1"""
    annotations_dir = Path(data_path) / 'annotations'
    all_files = sorted([f.stem for f in annotations_dir.glob('*.xml')])
    random.seed(seed)
    random.shuffle(all_files)
    
    n = len(all_files)
    n_train = int(0.75 * n)
    n_val = int(0.15 * n)
    test_files = all_files[n_train + n_val:]
    return test_files


def parse_annotation(xml_path):
    """Parse ground truth boxes from XML, return as [x, y, w, h]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        # convert to [x, y, w, h]
        boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
    
    return boxes