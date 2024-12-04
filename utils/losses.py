import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np
import config  # Custom configuration file

class NCCLoss:
    """
    Local (over window) Normalized Cross-Correlation (NCC) loss.
    This is commonly used for image or volume registration tasks.
    """

    def __init__(self, win=None):
        """
        Initializes the NCC loss with a given window size.
        
        Parameters:
            win (tuple or list, optional): The size of the window to compute the local NCC. 
                                            Defaults to a 9x9x9 window for 3D inputs.
        """
        self.win = win

    def loss(self, y_true, y_pred):
        """
        Computes the NCC loss between the true and predicted volumes.

        Parameters:
            y_true (torch.Tensor): The ground truth tensor with shape [batch_size, *vol_shape, nb_feats].
            y_pred (torch.Tensor): The predicted tensor with shape [batch_size, *vol_shape, nb_feats].

        Returns:
            torch.Tensor: The computed NCC loss value.
        """
        
        Ii = y_true
        Ji = y_pred

        # Get the number of dimensions (1D, 2D, or 3D)
        ndims = len(Ii.size()) - 2
        assert ndims in [1, 2, 3], f"Volumes should be 1 to 3 dimensions. Found: {ndims}"

        # Set default window size if not provided
        win = [9] * ndims if self.win is None else self.win

        # Compute the sum filter (kernel) for convolution
        sum_filt = torch.ones([1, 1, *win], device=Ii.device)

        # Calculate padding and stride based on dimensions
        pad_no = math.floor(win[0] / 2)
        stride = (1,) * ndims
        padding = (pad_no,) * ndims

        # Perform the convolution operation to compute local sums
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = F.conv3d(Ii, sum_filt, stride=stride, padding=padding) if ndims == 3 else \
                F.conv2d(Ii, sum_filt, stride=stride, padding=padding) if ndims == 2 else \
                F.conv1d(Ii, sum_filt, stride=stride, padding=padding)

        J_sum = F.conv3d(Ji, sum_filt, stride=stride, padding=padding) if ndims == 3 else \
                F.conv2d(Ji, sum_filt, stride=stride, padding=padding) if ndims == 2 else \
                F.conv1d(Ji, sum_filt, stride=stride, padding=padding)

        I2_sum = F.conv3d(I2, sum_filt, stride=stride, padding=padding) if ndims == 3 else \
                 F.conv2d(I2, sum_filt, stride=stride, padding=padding) if ndims == 2 else \
                 F.conv1d(I2, sum_filt, stride=stride, padding=padding)

        J2_sum = F.conv3d(J2, sum_filt, stride=stride, padding=padding) if ndims == 3 else \
                 F.conv2d(J2, sum_filt, stride=stride, padding=padding) if ndims == 2 else \
                 F.conv1d(J2, sum_filt, stride=stride, padding=padding)

        IJ_sum = F.conv3d(IJ, sum_filt, stride=stride, padding=padding) if ndims == 3 else \
                 F.conv2d(IJ, sum_filt, stride=stride, padding=padding) if ndims == 2 else \
                 F.conv1d(IJ, sum_filt, stride=stride, padding=padding)

        # Compute mean values over the window
        win_size = torch.prod(torch.tensor(win, device=Ii.device)).item()
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # Compute cross-correlation and variance
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute normalized cross-correlation (NCC)
        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1 - torch.mean(cc)

# Define the Dice3DMultiClass class, which inherits from the torch.nn.Module class
class Dice3DMultiClass(torch.nn.Module):
    '''
    Computes the Dice loss for multi-class segmentation tasks in 3D medical imaging.

    Parameters
    ----------
    num_classes : int
        Number of classes in the segmentation task.
    smooth : float, optional
        Smoothing factor to avoid division by zero, default is 1e-5.
    '''
    # Constructor method that initializes the class
    def __init__(self, num_classes, smooth=1e-5):
        '''
        Initialize the Dice3DMultiClass module.

        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        smooth : float, optional
            Smoothing factor to avoid division by zero, default is 1e-5.
        '''
        # Call the constructor of the parent class (torch.nn.Module)
        super(Dice3DMultiClass, self).__init__()
        
        # Set the number of classes and the smoothing factor as attributes
        self.num_classes = num_classes
        self.smooth = smooth

    # Forward method to compute the Dice loss during the forward pass
    def forward(self, prediction, target):
        '''
        Computes the Dice loss during the forward pass.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted probabilities tensor. Shape: [batch_size, num_classes, depth, height, width].
        target : torch.Tensor
            Ground truth segmentation mask tensor. Shape: [batch_size, num_classes, depth, height, width].

        Returns
        -------
        dice_losses : list of torch.Tensor
            List containing Dice loss for each class.
        '''
        # Convert the target to one-hot encoding (Note: This step may not be necessary depending on how the target is provided)
        target_one_hot = target

        # Initialize an empty list to store individual Dice losses for each class
        dice_losses = []

        # Iterate over each class
        for class_idx in range(self.num_classes):
            # Extract the predicted probabilities and corresponding one-hot encoded target for the current class
            class_prediction = prediction[:, class_idx, :, :, :]
            class_target = target_one_hot[:, class_idx, :, :, :]

            # Compute the intersection, union, and Dice coefficient for the current class
            intersection = torch.sum(class_prediction * class_target)
            union = torch.sum(class_prediction) + torch.sum(class_target)
            dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)
            # print(dice_coefficient)
            # Append the Dice loss to the list
            dice_losses.append(dice_coefficient)

            # break

        return dice_losses

class twinLoss:
    """
    Computes a custom loss function that combines invariance and redundancy losses.
    This is often used in contrastive learning methods such as Barlow Twins.
    """
    def __init__(self, lmbda):
        """
        Initializes the loss function with a given lambda scaling factor.
        
        Parameters:
            lmbda (float): The weight for the redundancy loss term.
        """
        super(twinLoss, self).__init__()
        self.lmbda = lmbda

    def __call__(self, z1, z2):
        """
        Computes the combined loss (invariance loss + redundancy loss) between 
        two normalized feature vectors (z1 and z2).
        
        Parameters:
            z1 (torch.Tensor): The first tensor of shape [batch_size, feature_dim].
            z2 (torch.Tensor): The second tensor of shape [batch_size, feature_dim].
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Normalize the projector's output across the batch (feature-wise normalization)
        norm_z1 = (z1 - z1.mean(0)) / z1.std(0)
        norm_z2 = (z2 - z2.mean(0)) / z2.std(0)

        # Cross-correlation matrix computation
        batch_size = z1.size(0)
        cc_M = torch.einsum('bi,bj->ij', (norm_z1, norm_z2)) / batch_size

        # Invariance loss: minimize off-diagonal entries and maximize diagonal (identity)
        diag = torch.diagonal(cc_M)
        invariance_loss = ((torch.ones_like(diag) - diag) ** 2).sum()

        # Redundancy loss: minimize cross-correlation between different examples (off-diagonal)
        cc_M.fill_diagonal_(0)  # Zero out the diagonal for redundancy computation
        redundancy_loss = (cc_M.flatten() ** 2).sum()

        # Total loss: combine invariance and redundancy losses
        loss = invariance_loss + self.lmbda * redundancy_loss

        return loss


class OptimalTransportLoss(nn.Module):
    """
    Computes the optimal transport loss between two sets of attention weights using a pairwise distance map.

    The loss is calculated by computing the weighted sum of distances between grid points for each set of attention weights.
    The final loss is the average of the individual costs for both sets of attention weights.

    Attributes:
        None

    Methods:
        forward: Computes the optimal transport loss given two sets of attention weights.
    """

    def __init__(self):
        """
        Initializes the OptimalTransportLoss module.
        """
        super(OptimalTransportLoss, self).__init__()

    def forward(self, Attention_weights_1, Attention_weights_2, device):
        """
        Forward pass for computing the optimal transport loss.

        Parameters:
            Attention_weights_1 (torch.Tensor): Attention weights for the first set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            Attention_weights_2 (torch.Tensor): Attention weights for the second set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            device (torch.device): The device (CPU/GPU) to perform the computations.

        Returns:
            torch.Tensor: The optimal transport loss, which is the average cost between the two sets of attention weights.
        """
        # Generate 3D grid coordinates for computing pairwise distances
        X, Y, Z = torch.meshgrid([torch.linspace(-1, 1, config.img_size//4), 
                                  torch.linspace(-1, 1, config.img_size//4), 
                                  torch.linspace(-1, 1, config.img_size//4)])
        coord = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).float()

        # Compute the pairwise distance map between all coordinates in the grid
        dist_map = torch.cdist(coord, coord).to(device)

        # Calculate the cost for the first set of attention weights based on pairwise distances
        Cost_Attention_weights_1 = torch.mean(torch.sum(Attention_weights_1.to(device) * dist_map, dim=(1, 2)))

        # Calculate the cost for the second set of attention weights based on pairwise distances
        Cost_Attention_weights_2 = torch.mean(torch.sum(Attention_weights_2.to(device) * dist_map, dim=(1, 2)))

        # Calculate the average cost between the two attention weights
        average_cost = (Cost_Attention_weights_1 + Cost_Attention_weights_2) / 2.0

        return average_cost


class OptimalTransportLoss_A1A2(nn.Module):
    """
    Computes the optimal transport loss between four sets of attention weights using a pairwise distance map.

    The loss is calculated by computing the weighted sum of distances between grid points for each set of attention weights.
    The final loss is the average of the individual costs for all four sets of attention weights.

    Attributes:
        None

    Methods:
        forward: Computes the optimal transport loss given four sets of attention weights.
    """

    def __init__(self):
        """
        Initializes the OptimalTransportLoss_A1A2 module.
        """
        super(OptimalTransportLoss_A1A2, self).__init__()

    def forward(self, Attention_weights_1, Attention_weights_2, Attention_weights_3, Attention_weights_4, device):
        """
        Forward pass for computing the optimal transport loss.

        Parameters:
            Attention_weights_1 (torch.Tensor): Attention weights for the first set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            Attention_weights_2 (torch.Tensor): Attention weights for the second set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            Attention_weights_3 (torch.Tensor): Attention weights for the third set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            Attention_weights_4 (torch.Tensor): Attention weights for the fourth set of features.
                                                   Shape: (batch_size, features, depth, height, width)
            device (torch.device): The device (CPU/GPU) to perform the computations.

        Returns:
            torch.Tensor: The optimal transport loss, which is the average cost between the four sets of attention weights.
        """
        # Generate 3D grid coordinates for computing pairwise distances
        X, Y, Z = torch.meshgrid([torch.linspace(-1, 1, 16), 
                                  torch.linspace(-1, 1, 16), 
                                  torch.linspace(-1, 1, 16)])
        coord = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).float()

        # Compute the pairwise distance map between all coordinates in the grid
        dist_map = torch.cdist(coord, coord).to(device)

        # Calculate the cost for each set of attention weights based on pairwise distances
        Cost_Attention_weights_1 = torch.mean(torch.sum(Attention_weights_1.to(device) * dist_map, dim=(1, 2)))
        Cost_Attention_weights_2 = torch.mean(torch.sum(Attention_weights_2.to(device) * dist_map, dim=(1, 2)))
        Cost_Attention_weights_3 = torch.mean(torch.sum(Attention_weights_3.to(device) * dist_map, dim=(1, 2)))
        Cost_Attention_weights_4 = torch.mean(torch.sum(Attention_weights_4.to(device) * dist_map, dim=(1, 2)))

        # Calculate the average cost across all four sets of attention weights
        average_cost = (Cost_Attention_weights_1 + Cost_Attention_weights_2 + 
                        Cost_Attention_weights_3 + Cost_Attention_weights_4) / 4.0

        return average_cost


def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.

    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.

    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis):
        return np.diff(array, axis=axis)[:, : (H - 1), : (W - 1), : (D - 1)]

    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = (
        dx[0] * (dy[1] * dz[2] - dz[1] * dy[2])
        - dy[0] * (dx[1] * dz[2] - dz[1] * dx[2])
        + dz[0] * (dx[1] * dy[2] - dy[1] * dx[2])
    )

    return det

def NoP_detJ(displacement_field):
    """
    Computes the fraction of non-positive Jacobian determinants in the displacement field.

    displacement_field is expected to be a tensor (4D), where the first dimension represents the displacement
    components (3, for x, y, z), and the remaining dimensions represent the spatial dimensions (H, W, D).
    
    Returns the fraction of non-positive Jacobian determinants.
    """
    # Convert displacement field to a numpy array and remove the first singleton dimension
    displacement_field = displacement_field.squeeze(0).to('cpu').detach().numpy()  # Convert to numpy (H, W, D, 3)
    
    # Compute the Jacobian determinant for the displacement field
    jacobian_determinant_ = jacobian_determinant(displacement_field)
    
    # Count the number of non-positive Jacobian determinant values
    num_non_positive = np.sum(jacobian_determinant_ < 0)  # Sum all negative determinant values
    
    # Compute the fraction of non-positive determinants
    num_non_positive_rates = num_non_positive / np.prod(jacobian_determinant_.shape)  # Fraction of non-positive values
    
    return num_non_positive_rates  # Return the computed fraction
