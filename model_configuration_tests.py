#1. Imports
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from data.datasets import make_loaders, ColorizationDataset
from models.conv_autoencoder import ColorizationNet

# 2. Reproducibility Settings
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "data/face_images"
dataset = ColorizationDataset(root_dir=path)

base_config = dict(epochs=30, batch_size=16, learning_rate=5e-4, weight_decay=1e-4, criterion="MSELoss")

configs = [
    # Comparison 1: Loss Function
    {**base_config, "criterion": "MSELoss"},
    {**base_config, "criterion": "SmoothL1Loss"},

    # Comparison 2: Regularization (Weight Decay)
    {**base_config, "weight_decay": 1e-4},
    {**base_config, "weight_decay": 1e-3},

    # Comparison 3: Learning Rate
    {**base_config, "learning_rate": 5e-4},
    {**base_config, "learning_rate": 1e-3},
]
for cfg in configs:
    with wandb.init(project="Neural_Networks_Project_UAB", config=cfg):
        config = wandb.config

        train_loader, val_loader, test_loader = make_loaders(dataset, config['batch_size'])

        model = ColorizationNet().to(device)
        print(model)

        if config['criterion'] == "MSELoss":
            criterion = nn.MSELoss()
        else:
            criterion = nn.SmoothL1Loss()

        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        model.train_model(train_loader, val_loader, config['epochs'], criterion, optimizer, scheduler, device)
        model.test_model(test_loader, device)