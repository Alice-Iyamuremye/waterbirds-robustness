"""ImageNet-pretrained ResNet-18 with a two-class Waterbirds head."""
import torch.nn as nn
from torchvision import models
from erm.config import MODEL_NAME,NUM_CLASSES,PRETRAINED

def build_model():
    if MODEL_NAME!="resnet18": raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
    weights=models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
    model=models.resnet18(weights=weights); model.fc=nn.Linear(model.fc.in_features,NUM_CLASSES)
    return model

def get_gradcam_target_layer(model): return model.layer4
