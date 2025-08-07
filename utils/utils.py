import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from skimage.color import lab2rgb, deltaE_cie76
from skimage.metrics import structural_similarity as ssim


def reconstruct_rgb(L_tensor, ab_tensor):
    """
    Converts a LAB image (from model output) into RGB.

    Args:
        L_tensor (Tensor): Grayscale L channel (normalized in [0, 1]).
        ab_tensor (Tensor): Predicted ab channels (normalized in [-1, 1]).

    Returns:
        np.ndarray: RGB image (in range [0, 1]).
    """
    L = L_tensor[0, 0].cpu().numpy() * 100
    ab = ab_tensor[0].cpu().numpy().transpose(1, 2, 0) * 128

    lab = np.zeros((L.shape[0], L.shape[1], 3))
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab

    rgb = lab2rgb(lab)
    return rgb


def evaluate_mse(ab_pred, ab_true):
    """
    Computes the Mean Squared Error between predicted and true ab channels.

    Args:
        ab_pred (Tensor): Predicted ab values.
        ab_true (Tensor): Ground truth ab values.

    Returns:
        float: MSE score.
    """
    mse = F.mse_loss(ab_pred, ab_true).item()
    return mse


def evaluate_psnr(ab_pred, ab_true):
    """
    Computes the PSNR between predicted and true ab channels.

    Args:
        ab_pred (Tensor): Predicted ab values.
        ab_true (Tensor): Ground truth ab values.

    Returns:
        float: PSNR in dB.
    """
    ab_pred_np = ab_pred.cpu().detach().numpy()[0]
    ab_true_np = ab_true.cpu().detach().numpy()[0]

    mse = np.mean((ab_pred_np - ab_true_np) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def evaluate_ssim(L, ab_pred, ab_true):
    """
    Computes the SSIM between predicted and ground truth RGB images.

    Args:
        L (Tensor): Input grayscale tensor.
        ab_pred (Tensor): Predicted ab channels.
        ab_true (Tensor): Ground truth ab channels.

    Returns:
        float: SSIM score in [0, 1].
    """
    pred_rgb = reconstruct_rgb(L, ab_pred)
    true_rgb = reconstruct_rgb(L, ab_true)

    ssim_score = ssim(pred_rgb, true_rgb, channel_axis=2, data_range=1.0)
    return ssim_score


def evaluate_delta_e(L, ab_pred, ab_true):
    """
    Computes the average Delta E (CIE76) color difference in LAB space.

    Args:
        L (Tensor): Grayscale input.
        ab_pred (Tensor): Predicted ab values.
        ab_true (Tensor): Ground truth ab values.

    Returns:
        float: Mean Delta E across all pixels.
    """
    H, W = L.shape[2], L.shape[3]

    pred_lab = np.zeros((H, W, 3))
    true_lab = np.zeros((H, W, 3))

    pred_lab[:, :, 0] = L.cpu().numpy()[0, 0] * 100
    true_lab[:, :, 0] = L.cpu().numpy()[0, 0] * 100

    pred_ab = ab_pred.cpu().detach().numpy()[0] * 128
    true_ab = ab_true.cpu().detach().numpy()[0] * 128

    pred_lab[:, :, 1:] = pred_ab.transpose(1, 2, 0)
    true_lab[:, :, 1:] = true_ab.transpose(1, 2, 0)

    delta_e = deltaE_cie76(pred_lab, true_lab)
    return np.mean(delta_e)


def show_result(L, ab_pred, ab_true):
    """
    Displays grayscale input, predicted RGB, and ground truth RGB images side by side.

    Args:
        L (Tensor): L channel (grayscale input).
        ab_pred (Tensor): Predicted ab channels.
        ab_true (Tensor): Ground truth ab channels.
    """
    rgb_pred = reconstruct_rgb(L, ab_pred)
    rgb_true = reconstruct_rgb(L, ab_true)

    plt.figure(figsize=(12, 4))

    # B&W input
    plt.subplot(1, 3, 1)
    plt.imshow(L.cpu().numpy()[0, 0], cmap='gray')
    plt.title("Input L (Grayscale)")
    plt.axis('off')

    # Predicted colorization
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_pred)
    plt.title("Predicted RGB")
    plt.axis('off')

    # Ground-truth color
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_true)
    plt.title("Original RGB")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_lab(L_img, ab_true, ab_pred, idx=None):
    """
    Displays a detailed comparison of ab channels and RGB reconstruction.

    Args:
        L_img (Tensor): Input L channel.
        ab_true (Tensor): Ground truth ab channels.
        ab_pred (Tensor): Predicted ab channels.
        idx (int, optional): Image index for console print.
    """
    ab_gt_np = ab_true[0].cpu().numpy().transpose(1, 2, 0)
    ab_pr_np = ab_pred[0].cpu().numpy().transpose(1, 2, 0)

    if idx is not None:
        print(f"\nTest Image {idx + 1}:")
    print(f"  Ground truth a: min={ab_gt_np[:, :, 0].min():.2f}, max={ab_gt_np[:, :, 0].max():.2f}")
    print(f"  Predicted     a: min={ab_pr_np[:, :, 0].min():.2f}, max={ab_pr_np[:, :, 0].max():.2f}")
    print(f"  Ground truth b: min={ab_gt_np[:, :, 1].min():.2f}, max={ab_gt_np[:, :, 1].max():.2f}")
    print(f"  Predicted     b: min={ab_pr_np[:, :, 1].min():.2f}, max={ab_pr_np[:, :, 1].max():.2f}")

    rgb_pred = reconstruct_rgb(L_img, ab_pred)
    rgb_gt = reconstruct_rgb(L_img, ab_true)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 6, 1)
    plt.imshow(L_img[0, 0].cpu().numpy(), cmap='gray')
    plt.title("Input L")
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(ab_gt_np[:, :, 0], cmap='RdYlGn')
    plt.title("GT a")
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(ab_pr_np[:, :, 0], cmap='RdYlGn')
    plt.title("Pred a")
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(ab_gt_np[:, :, 1], cmap='PuOr')
    plt.title("GT b")
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(ab_pr_np[:, :, 1], cmap='PuOr')
    plt.title("Pred b")
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(rgb_pred)
    plt.title("Predicted RGB")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
