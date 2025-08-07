import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from utils.utils import show_result, evaluate_mse, evaluate_psnr, evaluate_ssim, evaluate_delta_e, show_lab


class ColorizationNet(nn.Module):
    """
    Convolutional Autoencoder for image colorization.

    Takes a grayscale L channel image and outputs the corresponding ab color channels
    in the CIELAB color space. Architecture includes skip connections for feature reuse.
    """
    def __init__(self):
        super(ColorizationNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input grayscale image with shape [B, 1, H, W]

        Returns:
            Tensor: Output ab color channels, shape [B, 2, H, W]
        """
        x1 = self.enc1(x)       # 256x256
        x2 = self.enc2(x1)      # 128x128
        x3 = self.enc3(x2)      # 64x64
        x4 = self.enc4(x3)      # 32x32
        x5 = self.bottleneck(x4)

        u1 = self.up1(x5)       # 64x64
        u1 = torch.cat([u1, x3], dim=1)
        u2 = self.up2(u1)       # 128x128
        u2 = torch.cat([u2, x2], dim=1)
        u3 = self.up3(u2)       # 256x256
        u3 = torch.cat([u3, x1], dim=1)

        return self.final(u3)

    def train_model(self, train_loader, test_loader, epochs, criterion, optimizer, scheduler=None, device="cuda"):
        """
        Trains the autoencoder model.

        Args:
            train_loader (DataLoader): Training set loader.
            test_loader (DataLoader): Validation set loader.
            epochs (int): Number of training epochs.
            criterion (Loss): Loss function (e.g., MSELoss or SmoothL1Loss).
            optimizer (Optimizer): Optimizer (e.g., AdamW).
            scheduler (optional): Learning rate scheduler.
            device (str): 'cuda' or 'cpu'.
        """
        wandb.watch(self, criterion, log="all", log_freq=10)
        train_losses = []
        val_losses = []

        # Training
        for epoch in range(epochs):
            self.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
                L = batch['L'].to(device)
                ab = batch['ab'].to(device)

                pred = self(L)
                loss = criterion(pred, ab)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} validation"):
                    L = batch['L'].to(device)
                    ab = batch['ab'].to(device)
                    pred = self(L)
                    val_loss = criterion(pred, ab)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            # Scheduler and logging
            if scheduler:
                scheduler.step(avg_val_loss)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    def test_model(self, test_loader, device):
        """
        Evaluates the trained model on the test set and logs metrics.

        Args:
            test_loader (DataLoader): Test set loader.
            device (str): 'cuda' or 'cpu'
        """
        self.eval()
        mse_total = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        delta_e_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                L = batch['L'].to(device)
                ab_true = batch['ab'].to(device)

                ab_pred = self(L)

                # Show the first 3 images
                if num_batches < 3:
                    show_result(L, ab_pred, ab_true)
                    show_lab(L, ab_true, ab_pred, idx=num_batches)

                # Compute metrics
                mse_score = evaluate_mse(ab_pred, ab_true)
                psnr_score = evaluate_psnr(ab_pred, ab_true)
                ssim_score = evaluate_ssim(L, ab_pred, ab_true)
                delta_e_score = evaluate_delta_e(L, ab_pred, ab_true)

                mse_total += mse_score
                psnr_total += psnr_score
                ssim_total += ssim_score
                delta_e_total += delta_e_score
                num_batches += 1

        # Compute averages
        avg_mse = mse_total / num_batches
        avg_psnr = psnr_total / num_batches
        avg_ssim = ssim_total / num_batches
        avg_delta_e = delta_e_total / num_batches

        print(f"Average MSE on test set:     {avg_mse:.4f}")
        print(f"Average PSNR on test set:    {avg_psnr:.2f} dB")
        print(f"Average SSIM on test set:    {avg_ssim:.4f}")
        print(f"Average Delta E on test set: {avg_delta_e:.4f}")

        # Log metrics to Weights & Biases
        wandb.log({
            "test/mse": avg_mse,
            "test/psnr": avg_psnr,
            "test/ssim": avg_ssim,
            "test/delta_e": avg_delta_e
        })
