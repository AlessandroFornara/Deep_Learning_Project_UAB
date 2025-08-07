import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from utils.utils import show_result, evaluate_mse, evaluate_psnr, evaluate_ssim, evaluate_delta_e, show_lab


class Generator(nn.Module):
    """
    Convolutional Generator for image colorization.

    Takes a single-channel L (grayscale) image and predicts
    the two-channel ab (color) components in the LAB color space.
    """
    def __init__(self):
        super().__init__()

        # Encoder: progressively downsamples while increasing channel depth
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Decoder: upsamples and reconstructs the ab output
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1], matching normalized LAB ab channels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator for distinguishing real vs fake LAB image pairs.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, padding=1),
            nn.Dropout(0.2),
            nn.Sigmoid()  # Output: [B, 1, H/8, W/8]
        )

    def forward(self, x):
        return self.net(x)


class AdvModel:
    """
    Wrapper class for training and testing a GAN-based image colorization model.
    """
    def __init__(self, generator, discriminator):
        """
        Args:
            generator (nn.Module): The generator network.
            discriminator (nn.Module): The discriminator network.
        """
        self.generator = generator
        self.discriminator = discriminator

    def train_model(self, train_loader,
                    d_criterion, g_loss_fn,
                    g_optimizer, d_optimizer,
                    lambda_l1=100,
                    epochs=20,
                    device="cuda"):
        """
        Trains the GAN model using adversarial and L1 losses.

        Args:
            train_loader (DataLoader): Training data loader.
            d_criterion (Loss): Discriminator loss (e.g., BCE).
            g_loss_fn (Loss): Generator reconstruction loss (e.g., L1).
            g_optimizer (Optimizer): Optimizer for generator.
            d_optimizer (Optimizer): Optimizer for discriminator.
            lambda_l1 (float): Weight of L1 loss term.
            epochs (int): Number of training epochs.
            device (str): 'cuda' or 'cpu'.
        """
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()

            total_d_loss = 0.0
            total_g_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                L = batch['L'].to(device)
                ab = batch['ab'].to(device)

                # Discriminator
                real_pair = torch.cat([L, ab], dim=1)  # (B, 3, H, W)
                fake_ab = self.generator(L).detach()
                fake_pair = torch.cat([L, fake_ab], dim=1)

                real_label = torch.ones((L.size(0), 1, 30, 30), device=device)
                fake_label = torch.zeros_like(real_label)

                # Forward pass through discriminator
                d_real = self.discriminator(real_pair)
                d_fake = self.discriminator(fake_pair)

                # Dynamically create labels
                real_label = torch.ones_like(d_real)
                fake_label = torch.zeros_like(d_fake)

                # Losses
                d_loss_real = d_criterion(d_real, real_label)
                d_loss_fake = d_criterion(d_fake, fake_label)
                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                fake_ab = self.generator(L)
                fake_pair = torch.cat([L, fake_ab], dim=1)
                pred_fake = self.discriminator(fake_pair)

                adv_loss = d_criterion(pred_fake, real_label)
                l1 = g_loss_fn(fake_ab, ab)
                g_loss = adv_loss + lambda_l1 * l1

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                num_batches += 1

            # Log average losses to Weights & Biases
            avg_d_loss = total_d_loss / num_batches
            avg_g_loss = total_g_loss / num_batches

            wandb.log({
                "epoch": epoch + 1,
                "discriminator_loss": avg_d_loss,
                "generator_loss": avg_g_loss,
            })

            print(f"Epoch {epoch + 1} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    def test_model(self, test_loader, device):
        """
        Evaluates the generator on test data and logs qualitative + quantitative results.

        Args:
            test_loader (DataLoader): Test data loader.
            device (str): 'cuda' or 'cpu'.
        """
        self.generator.eval()

        mse_total = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        delta_e_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                L = batch['L'].to(device)
                ab_true = batch['ab'].to(device)
                ab_pred = self.generator(L)

                # Show the first 3 images
                if i < 3:
                    show_result(L, ab_pred, ab_true)
                    show_lab(L, ab_true, ab_pred, idx=i)

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
