#1. Imports
import random
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from data.datasets import make_loaders, ColorizationDataset, download_landscape_dataset
from models.adv_generator import Generator, Discriminator, AdvModel
from models.conv_autoencoder import ColorizationNet

# 2. Reproducibility Settings
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelType(Enum):
    AUTOENCODER = "Conv_Autoencoder"
    GAN = "Gen_Adv"


class DatasetType(Enum):
    FACES = "Faces"
    LANDSCAPES = "Landscapes"


class LossesType(Enum):
    MSE = "MSE"
    BCE = "BCE"
    L1 = "L1"


if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=30,
        batch_size=20,
        learning_rate=5e-4,
        weight_decay=1e-4,
        criterion=LossesType.MSE.value,
        dataset=DatasetType.FACES.value,
        architecture=ModelType.AUTOENCODER.value
    )

    with wandb.init(project="Neural_Networks_Project_UAB", config=config):
        cfg = wandb.config

        # Load dataset
        if cfg.dataset == DatasetType.FACES.value:
            path = "data/face_images"
        elif cfg.dataset == DatasetType.LANDSCAPES.value:
            path = download_landscape_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {cfg.dataset}")

        dataset = ColorizationDataset(root_dir=path)

        train_loader, val_loader, test_loader = make_loaders(dataset, cfg.batch_size)

        # CONVOLUTIONAL AUTOENCODER
        if cfg.architecture == ModelType.AUTOENCODER.value:
            model = ColorizationNet().to(device)
            print(model)

            if cfg.criterion == LossesType.MSE.value:
                print("Selected MSE")
                criterion = nn.MSELoss()
            elif cfg.criterion == LossesType.L1.value:
                print("Selected SmoothL1")
                criterion = nn.SmoothL1Loss()
            else:
                raise ValueError(f"Unsupported loss for this model: {cfg.criterion}")

            optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            model.train_model(train_loader, val_loader, cfg.epochs, criterion, optimizer, scheduler, device)
            model.test_model(test_loader, device)

        # ADV GENERATOR
        elif cfg.architecture == ModelType.GAN.value:
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            model = AdvModel(generator, discriminator)

            bce_loss = nn.BCELoss()
            l1_loss = nn.L1Loss()
            g_optimizer = optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
            d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

            model.train_model(train_loader, bce_loss, l1_loss, g_optimizer, d_optimizer, lambda_l1=100,
                              epochs=cfg.epochs, device=device)
            model.test_model(test_loader, device)

        else:
            raise ValueError(f"Unknown model type: {cfg.architecture}")
