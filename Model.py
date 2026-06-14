# models/efficientnet_b3.py

import torch.nn as nn
import torchvision.models as models


def build_model(num_classes):

    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model
