import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ECGResNet(nn.Module):
    """
    ResNet50 fine-tuned for 5-class CAC score classification from ECG PNG images.
 
    Strategy:
        - Freeze early ResNet layers (layers 1-3) to preserve ImageNet features
        - Fine-tune only layer4 + classifier head
        - Strong dropout in head to prevent overfitting on small medical datasets
    """
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
 
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
 
        # Freeze everything except layer4
        for name, param in resnet.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False
 
        in_features = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()
        self.backbone = resnet
 
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
 
    def _get_name(self):
        return "ECGResNet50"
 
    def forward(self, x):
        features = self.backbone(x)        
        return self.classifier(features) 
 

 # Model 2: EfficientNet-B0 fine-tuned
class ECGEfficientNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class CAC classification.
    Lighter than ResNet50, good for small datasets.
    """
    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()
        
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze early layers
        for name, param in efficientnet.named_parameters():
            if not name.startswith("features.7") and not name.startswith("features.8") and not name.startswith("classifier"):
                param.requires_grad = False
        
        in_features = efficientnet.classifier[1].in_features  # 1280
        efficientnet.classifier = nn.Identity()
        self.backbone = efficientnet
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def _get_name(self):
        return "ECGEfficientNetB0"
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Model 3: MobileNetV3 fine-tuned
class ECGMobileNet(nn.Module):
    """
    MobileNetV3-Small fine-tuned for 5-class CAC classification.
    Very lightweight, fast inference.
    """
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Freeze early features
        for name, param in mobilenet.named_parameters():
            if not name.startswith("features.11") and not name.startswith("features.12") and not name.startswith("classifier"):
                param.requires_grad = False
        
        in_features = mobilenet.classifier[0].in_features  # 576
        mobilenet.classifier = nn.Identity()
        self.backbone = mobilenet
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def _get_name(self):
        return "ECGMobileNetV3"
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Model 4: DenseNet121 fine-tuned
class ECGDenseNet(nn.Module):
    """
    DenseNet121 fine-tuned for 5-class CAC classification.
    Dense connections help with gradient flow on small datasets.
    """
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Freeze early layers
        for name, param in densenet.named_parameters():
            if not name.startswith("features.denseblock4") and not name.startswith("features.norm5") and not name.startswith("classifier"):
                param.requires_grad = False
        
        in_features = densenet.classifier.in_features  # 1024
        densenet.classifier = nn.Identity()
        self.backbone = densenet
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def _get_name(self):
        return "ECGDenseNet121"
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)