import os

import kagglehub
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for image colorization tasks.

    Loads RGB images from a directory, converts them to LAB color space,
    and returns the L (grayscale) channel as input and ab channels as target.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with .jpg images.
            transform (callable, optional): Transform to be applied on each image.
        """
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith('.jpg')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and convert image to RGB
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Apply transformations (resize, crop, etc.)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

         # === ORIGINAL LAB L VERSION ===
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for LAB
        lab = rgb2lab(img).astype("float32")
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:] / 128.0
        L = np.expand_dims(L, axis=0)
        ab = np.transpose(ab, (2, 0, 1))

        # === STANDARD GRAYSCALE VERSION ===
        #gray_img = transforms.functional.rgb_to_grayscale(img, num_output_channels=1)  # shape: 1 x H x W
        #L = gray_img.numpy()  # Already in shape [1, H, W], values in [0, 1]

        #img = img.permute(1, 2, 0).numpy()  # Convert to HWC format again for LAB
        #lab = rgb2lab(img).astype("float32")
        #ab = lab[:, :, 1:] / 128.0
        #ab = np.transpose(ab, (2, 0, 1))


        return {
            'L': torch.tensor(L, dtype=torch.float32),
            'ab': torch.tensor(ab, dtype=torch.float32)
        }


def make_loaders(dataset, batch_size):
    """
    Splits dataset into train/val/test and returns corresponding DataLoaders.

    Args:
        dataset (Dataset): The full dataset to split.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define augmentations and basic transforms
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    # Split dataset: 70% train, 15% val, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

    # Assign transforms to subsets
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_test_transforms
    test_subset.dataset.transform = val_test_transforms

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def download_landscape_dataset():
    """
    Downloads the landscape image colorization dataset from Kaggle Hub.

    Returns:
        str: Path to the 'color' images inside the dataset.
    """
    # Download dataset
    path = kagglehub.dataset_download("theblackmamba31/landscape-image-colorization")
    print("Path to dataset files:", path)

    color_image_path = os.path.join(path, "landscape Images", "color")
    return color_image_path
