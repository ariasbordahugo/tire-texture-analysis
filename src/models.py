import torch
import torch.nn as nn
import torchvision.models as models

class CustomTextureCNN(nn.Module):
    """
    Arquitectura Convolucional Baseline construida desde cero
    para la extracción de texturas industriales.
    """
    def __init__(self, num_classes=2):
        super(CustomTextureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TransferLearningResNet(nn.Module):
    """
    Arquitectura ResNet-50 aplicando Transfer Learning
    con pesos de ImageNet para detección de anomalías.
    """
    def __init__(self, num_classes=2):
        super(TransferLearningResNet, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Congelar el backbone (Feature Extractor)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Reemplazar la cabeza de clasificación
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)
