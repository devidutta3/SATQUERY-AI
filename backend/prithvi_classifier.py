import torch
import torch.nn as nn


class PrithviClassifier(nn.Module):

    def __init__(self, feature_dim=1024, num_classes=5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, features):

        return self.classifier(features)
