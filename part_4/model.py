"""
Part 4: Faster R-CNN model with configurable backbone and anchors
"""
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

def get_model(num_classes=2, pretrained=True, backbone='resnet50', small_anchors=False):
    """
    Load Faster R-CNN with configurable backbone.
    
    Args:
        num_classes: 2 for pothole detection (bg + pothole)
        pretrained: use imagenet pretrained weights
        backbone: 'resnet50' or 'mobilenet'
        small_anchors: use smaller anchor sizes (better for small objects)
    """
    # load base model with new weights API
    if backbone == 'resnet50':
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
    elif backbone == 'mobilenet':
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # custom anchors for detecting smaller potholes
    if small_anchors:
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        model.rpn.anchor_generator = anchor_generator

    # replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    configs = [
        {'backbone': 'resnet50', 'small_anchors': False},
        {'backbone': 'resnet50', 'small_anchors': True},
        {'backbone': 'mobilenet', 'small_anchors': False},
    ]

    for cfg in configs:
        model = get_model(num_classes=2, **cfg)
        total = sum(p.numel() for p in model.parameters())
        print(f"{cfg['backbone']} (small_anchors={cfg['small_anchors']}): {total:,} params")