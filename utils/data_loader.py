"""
Custom Dataset for Loading and Processing NIfTI Images and Masks

This script defines functions for reading NIfTI images and masks, 
normalizing them, and converting them into PyTorch tensors. 
It also implements a custom dataset class for handling paired medical images and masks 
using MONAI's Dataset framework.

Authors:
- Md Kamrul Hasan

Date:
- 19-Nov-2024

Dependencies:
- PyTorch
- MONAI
- nibabel
"""

# Import necessary libraries
import torch
import os
from monai.data import DataLoader, Dataset  # MONAI dataset utilities
import nibabel as nib  # Library for handling NIfTI files


def readNifti_img(path):
    """
    Reads a NIfTI image from the specified path, normalizes pixel values, and converts it to a torch tensor.

    Parameters
    ----------
    path : str
        Path to the NIfTI image file.

    Returns
    -------
    img_ : torch.Tensor
        Torch tensor representing the NIfTI image with an added channel dimension.
    """
    # Load the NIfTI image using nibabel
    img_ = nib.load(path).get_fdata()

    # Normalize pixel values to the range [0, 1]
    img_ = img_ / img_.max()

    # Convert to a torch tensor and add a channel dimension
    img_ = torch.tensor(img_, dtype=torch.float32).unsqueeze(0)

    return img_


def readNifti_mask(path):
    """
    Reads a NIfTI mask from the specified path and converts it to a torch tensor.

    Parameters
    ----------
    path : str
        Path to the NIfTI mask file.

    Returns
    -------
    mask_ : torch.Tensor
        Torch tensor representing the NIfTI mask with an added channel dimension.
    """
    # Load the NIfTI mask using nibabel
    mask_ = nib.load(path).get_fdata()

    # Convert to a torch tensor and add a channel dimension
    mask_ = torch.tensor(mask_, dtype=torch.float32).unsqueeze(0)

    return mask_


class NiftiDataset(Dataset):
    """
    Custom Dataset for paired medical images and masks using MONAI's Dataset framework.

    This dataset assumes paired 'fixed' and 'moving' images and masks for medical image registration tasks.
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Initializes the dataset with image and mask paths, and optional transformations.

        Parameters
        ----------
        image_paths : list
            List containing paths to the images.
        mask_paths : list
            List containing paths to the masks.
        transform : callable, optional
            A function or transformation to apply to each sample (default: None).
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing fixed and moving images and masks.
        """
        # Extract the directory and file name for the current image
        head, tail = os.path.split(self.image_paths[idx])

        # Load fixed and moving images
        FixedPath_img = readNifti_img(os.path.join(head, tail[:-9] + 'mving.nii'))
        MovingPath_img = readNifti_img(os.path.join(head, tail[:-9] + 'fixed.nii'))

        # Extract the directory and file name for the current mask
        head, tail = os.path.split(self.mask_paths[idx])

        # Load fixed and moving masks
        FixedPath_mask = readNifti_mask(os.path.join(head, tail[:-9] + 'mving.nii'))
        MovingPath_mask = readNifti_mask(os.path.join(head, tail[:-9] + 'fixed.nii'))

        # Create a dictionary with all relevant data
        subject = {
            'fixed_img': FixedPath_img,
            'fixed_mask': FixedPath_mask,
            'moving_img': MovingPath_img,
            'moving_mask': MovingPath_mask,
            'name': tail[:-9]  # Extract the base name of the image
        }

        # Apply transformations if provided
        if self.transform:
            subject = self.transform(subject)

        return subject
