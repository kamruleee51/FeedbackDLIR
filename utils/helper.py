# Importing necessary libraries
import torch  # PyTorch library for tensor operations
from torch.autograd import Variable  # For automatic differentiation
import matplotlib.pyplot as plt  # Library for plotting
import config  # Custom configuration file
import torch.nn as nn
import torch.nn.functional as F


# Function for plotting predictions
def imshows(fixed, moving): 
    '''
    Plots the predicted, fixed, and moving masks.

    Parameters
    ----------
    pred_mask_train : torch.Tensor
        Predicted mask tensor.
    fixed_train_msk : torch.Tensor
        Fixed mask tensor.
    moving_train_msk : torch.Tensor
        Moving mask tensor.
    '''
    plt.figure(figsize=(8, 3.5))

    fixed_ = (fixed[0][0]).reshape(config.img_size, config.img_size, config.img_size).detach().cpu().numpy()
    plt.subplot(1, 2, 1)
    plt.title("Fixed")
    plt.imshow(fixed_[:,:,config.img_size//2-0], cmap="gray")
    plt.axis('off')

    moving_ = (moving[0][0]).reshape(config.img_size, config.img_size, config.img_size).detach().cpu().numpy()
    plt.subplot(1, 2, 2)
    plt.title("Moving")
    plt.imshow(moving_[:,:,config.img_size//2-0], cmap="gray")
    plt.axis('off')


def estParams(model):
    """
    Compute the total number of trainable parameters in a given neural network model.

    This function iterates through all parameters in the model, calculates the product of dimensions
    for each parameter tensor, and sums them up to get the total number of parameters.

    Args:
        model (torch.nn.Module): The neural network model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    total_parameters = 0  # Initialize parameter counter
    
    # Iterate over all parameters in the model
    for parameter in model.parameters():
        i = len(parameter.size())  # Get the dimensionality of the parameter tensor
        p = 1  # Initialize product as 1
        
        # Compute the product of dimensions for the parameter tensor
        for j in range(i):
            p *= parameter.size(j)
        
        total_parameters += p  # Add to total parameter count

    return total_parameters

# Function to convert integer labels to one-hot encoding
def make_one_hot(labels, device, C=2):
    '''
    Converts integer labels to one-hot encoding for semantic segmentation tasks.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        Shape: N x 1 x H x W, where N is the batch size. 
        Each value is an integer representing correct classification.
    C : integer
        Number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        Shape: N x C x H x W, where C is the class number. One-hot encoded.
    '''
    # Ensure labels are of type LongTensor
    labels = labels.long()
    
    # Create a zero-initialized one-hot tensor with the appropriate dimensions
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3), labels.size(4)).zero_().to(device)
    
    # Use scatter_ to set the corresponding class index to 1 for each pixel
    target = one_hot.scatter_(1, labels.data, 1)
    
    # Convert the result to a torch.autograd.Variable
    target = Variable(target)
        
    return target


# Function for thresholding the prediction mask
def thresholded(predMask, LB, UB):
    '''
    Thresholds the prediction mask.

    Parameters
    ----------
    predMask : torch.Tensor
        Prediction mask tensor.
    LB : float
        Lower bound threshold value.
    UB : float
        Upper bound threshold value.

    Returns
    -------
    thresholded_predMask_ : torch.Tensor
        Thresholded prediction mask.
    '''
    # Thresholding the prediction
    thresholded_predMask = predMask.clone()

    background_class = torch.zeros_like(predMask)
    myo_class = torch.ones_like(predMask)
    lv_class = 2 * torch.ones_like(predMask)

    thresholded_predMask_ = torch.where(thresholded_predMask < LB, background_class, thresholded_predMask)
    thresholded_predMask_ = torch.where(thresholded_predMask_ > UB, lv_class, thresholded_predMask_)
    thresholded_predMask_ = torch.where((thresholded_predMask_ <= UB) & (thresholded_predMask_ >= LB), myo_class, thresholded_predMask_)

    return thresholded_predMask_