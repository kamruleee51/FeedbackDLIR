import torch.nn as nn
import torch
import torch.nn.functional as F
import config  # Custom configuration file

def Convolution_2x(in_channels, out_channels):
    """
    Creates a double convolutional block with 3D convolutions, group normalization, 
    and ReLU activations. This is a common building block for 3D U-Net architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the first convolutional layer.
    out_channels : int
        Number of output channels for the second convolutional layer.

    Returns
    -------
    nn.Sequential
        A sequential container with the following layers:
        - 3D Convolution: Reduces `in_channels` to `out_channels / 2`.
        - GroupNorm: Normalizes the output of the first convolution.
        - ReLU: Activation function for non-linearity.
        - 3D Convolution: Expands `out_channels / 2` to `out_channels`.
        - GroupNorm: Normalizes the output of the second convolution.
        - ReLU: Activation function for non-linearity.

    Notes
    -----
    - `GroupNorm` divides channels into groups for normalization, offering better 
      generalization compared to batch normalization in small batch scenarios.
    - `padding=1` ensures the output spatial dimensions remain the same as the input.
    """
    return nn.Sequential(
        nn.Conv3d(in_channels, int(out_channels / 2), kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=int(out_channels / 8), num_channels=int(out_channels / 2)),
        nn.ReLU(inplace=True),
        nn.Conv3d(int(out_channels / 2), out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=int(out_channels / 4), num_channels=out_channels),
        nn.ReLU(inplace=True)
    )


# Spatial Attention Block
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.attention_conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        avg_pool = F.avg_pool3d(input_tensor, kernel_size=input_tensor.size()[2:])
        max_pool = F.max_pool3d(input_tensor, kernel_size=input_tensor.size()[2:])
        combined_pool = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.sigmoid(self.attention_conv(combined_pool))
        return input_tensor * attention_map



# Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        avg_out = self.fc(self.avg_pool(input_tensor))
        max_out = self.fc(self.max_pool(input_tensor))
        attention_map = self.sigmoid(avg_out + max_out)
        return input_tensor * attention_map



# Residual Attention Block
class ResBlock_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock_Attention, self).__init__()
        self.double_conv = Convolution_2x(in_channels, out_channels)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, input_tensor):
        features = self.double_conv(input_tensor)
        identity = features
        features_with_channel_attention = self.channel_attention(features)
        features_with_spatial_attention = self.spatial_attention(features_with_channel_attention)
        return features_with_spatial_attention + identity  # Residual connection


class ASPP(nn.Module):
    def __init__(self, dilation_rates, padding_values, output_channels):
        """
        Atrous Spatial Pyramid Pooling (ASPP) implementation for 3D inputs.
        
        Parameters
        ----------
        dilation_rates : list
            List of dilation rates for the atrous convolutions.
        padding_values : list
            List of padding values corresponding to each dilation rate.
        output_channels : int
            Number of output channels for the ASPP module.
        """
        super(ASPP, self).__init__()

        self.global_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_pooling_conv = nn.Conv3d(256, output_channels, kernel_size=1)
        self.global_pooling_norm = nn.GroupNorm(num_groups=output_channels // 4, num_channels=output_channels)

        self.conv_1x1 = nn.Conv3d(256, output_channels, kernel_size=1, stride=1)
        self.conv_1x1_norm = nn.GroupNorm(num_groups=output_channels // 4, num_channels=output_channels)

        self.conv_3x3_dil1 = nn.Conv3d(256, output_channels, kernel_size=3, stride=1, 
                                       padding=padding_values[0], dilation=dilation_rates[0])
        self.conv_3x3_dil1_norm = nn.GroupNorm(num_groups=output_channels // 4, num_channels=output_channels)

        self.conv_3x3_dil2 = nn.Conv3d(256, output_channels, kernel_size=3, stride=1, 
                                       padding=padding_values[1], dilation=dilation_rates[1])
        self.conv_3x3_dil2_norm = nn.GroupNorm(num_groups=output_channels // 4, num_channels=output_channels)

        self.conv_3x3_dil3 = nn.Conv3d(256, output_channels, kernel_size=3, stride=1, 
                                       padding=padding_values[2], dilation=dilation_rates[2])
        self.conv_3x3_dil3_norm = nn.GroupNorm(num_groups=output_channels // 4, num_channels=output_channels)

        self.relu = nn.ReLU(inplace=True)

        self.bottleneck_conv = nn.Conv3d(output_channels * 5, 256, kernel_size=3, padding=1)
        self.bottleneck_norm = nn.GroupNorm(num_groups=256 // 4, num_channels=256)
        self.bottleneck_activation = nn.PReLU()

        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.out_channels
                layer.weight.data.normal_(0, 0.01)
            elif isinstance(layer, nn.GroupNorm):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass through the ASPP module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after applying ASPP.
        """
        input_size = x.shape[2:]

        # Global average pooling branch
        global_features = self.global_pooling(x)
        global_features = self.global_pooling_conv(global_features)
        global_features = self.global_pooling_norm(global_features)
        global_features = self.relu(global_features)
        global_features = F.upsample(global_features, size=input_size, mode='trilinear', align_corners=True)

        # 1x1 convolution branch
        conv_1x1 = self.conv_1x1(x)
        conv_1x1 = self.conv_1x1_norm(conv_1x1)
        conv_1x1 = self.relu(conv_1x1)

        # 3x3 convolution with dilation rate 1
        conv_3x3_1 = self.conv_3x3_dil1(x)
        conv_3x3_1 = self.conv_3x3_dil1_norm(conv_3x3_1)
        conv_3x3_1 = self.relu(conv_3x3_1)

        # 3x3 convolution with dilation rate 2
        conv_3x3_2 = self.conv_3x3_dil2(x)
        conv_3x3_2 = self.conv_3x3_dil2_norm(conv_3x3_2)
        conv_3x3_2 = self.relu(conv_3x3_2)

        # 3x3 convolution with dilation rate 3
        conv_3x3_3 = self.conv_3x3_dil3(x)
        conv_3x3_3 = self.conv_3x3_dil3_norm(conv_3x3_3)
        conv_3x3_3 = self.relu(conv_3x3_3)

        # Concatenate features from all branches
        concatenated_features = torch.cat([global_features, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3], dim=1)

        # Bottleneck layer to reduce dimensionality
        output = self.bottleneck_conv(concatenated_features)
        output = self.bottleneck_norm(output)
        output = self.bottleneck_activation(output)

        return output



class SpatialTransformer(nn.Module):
    """
    A 3D Spatial Transformer module for deformable image registration.

    Attributes
    ----------
    mode : str
        Interpolation mode for sampling (default: 'bilinear').
    grid : torch.Tensor
        A precomputed meshgrid of voxel coordinates for the spatial domain.
    sigmoid : nn.Sigmoid
        A sigmoid activation function (not directly used in this implementation but defined as a buffer).
    """

    def __init__(self, mode='bilinear'):
        """
        Initialize the SpatialTransformer module.

        Parameters
        ----------
        mode : str, optional
            Interpolation mode for grid sampling. Default is 'bilinear'.
        """
        super(SpatialTransformer, self).__init__()

        # Define the spatial grid size based on configuration
        size = [config.img_size, config.img_size, config.img_size]  # 3D grid size
        self.mode = mode

        # Create a grid of voxel coordinates
        vectors = [torch.arange(0, s) for s in size]
        mesh_1, mesh_2, mesh_3 = torch.meshgrid(vectors)  # Create 3D meshgrid
        grid = torch.stack((mesh_1, mesh_2, mesh_3), 3)  # Combine into (x, y, z)
        grid = grid.unsqueeze(0).float()  # Add batch dimension and convert to float

        # Register the grid as a persistent buffer (not a parameter)
        self.register_buffer('grid', grid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, flow):
        """
        Apply the spatial transformation to the source image using the deformation field.

        Parameters
        ----------
        src : torch.Tensor
            The source image tensor of shape (B, C, D, H, W).
        flow : torch.Tensor
            The deformation field of shape (B, 3, D, H, W).

        Returns
        -------
        torch.Tensor
            The transformed source image of the same shape as `src`.
        """
        # Get the shape of the deformation field
        shape = flow.shape[2:]  # Extract (D, H, W) dimensions

        # Permute the flow to align dimensions with the grid
        flow = flow.permute(0, 2, 3, 4, 1)  # Change to (B, D, H, W, 3)

        # Add the flow to the precomputed grid
        new_loc = self.grid + flow

        # Normalize the grid values to [-1, 1] for compatibility with grid_sample
        new_loc[:, :, :, :, 0] = 2 * (new_loc[:, :, :, :, 0] / (shape[0] - 1) - 0.5)
        new_loc[:, :, :, :, 1] = 2 * (new_loc[:, :, :, :, 1] / (shape[1] - 1) - 0.5)
        new_loc[:, :, :, :, 2] = 2 * (new_loc[:, :, :, :, 2] / (shape[2] - 1) - 0.5)

        # Flip the coordinate channels to match the expected format for grid_sample
        new_loc = new_loc[..., [2, 1, 0]]  # Change (x, y, z) -> (z, y, x)

        # Use grid_sample to resample the source image
        return F.grid_sample(src, new_loc, align_corners=True, mode=self.mode)


class encoder(nn.Module):
    def __init__(self, output_channels):
        '''
        Encoder block for extracting multi-scale features using successive convolutional layers and downsampling.

        Parameters
        ----------
        output_channels : int
            Number of output channels for the final convolutional layer.
        '''
        super(encoder, self).__init__()
        self.conv_block1 = Convolution_2x(1, 64)          # First convolutional block
        self.conv_block2 = Convolution_2x(64, 128)        # Second convolutional block
        self.conv_block3 = Convolution_2x(128, output_channels)  # Third convolutional block
        self.max_pooling = nn.MaxPool3d(kernel_size=2)    # 3D max pooling for downsampling

    def forward(self, input_tensor):
        '''
        Forward pass through the encoder block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Encoded feature tensor after the forward pass.
        '''
        features_level1 = self.conv_block1(input_tensor)   # First level of features
        downsampled_level1 = self.max_pooling(features_level1)  # Downsampling

        features_level2 = self.conv_block2(downsampled_level1)  # Second level of features
        downsampled_level2 = self.max_pooling(features_level2)  # Downsampling

        features_level3 = self.conv_block3(downsampled_level2)  # Final feature representation

        return features_level3


class MultiLayerPerceptron(nn.Module):
    '''
    A simple Multi-Layer Perceptron (MLP) for feature reduction.

    Parameters
    ----------
    dim : int, optional
        Output feature dimension, default is 128.
    '''
    def __init__(self, output_dim=128):
        super(MultiLayerPerceptron, self).__init__()
        self.flatten_dim = 256 * 16 * 16 * 16  # Flatten input dimension
        self.fc_layer1 = nn.Linear(self.flatten_dim, 512)  # First fully connected layer
        self.fc_layer2 = nn.Linear(512, 512)              # Second fully connected layer
        self.output_layer = nn.Linear(512, output_dim)    # Final output layer

    def forward(self, input_tensor):
        '''
        Forward pass of the MLP.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (batch_size, 256, 16, 16, 16).

        Returns
        -------
        torch.Tensor
            Reduced feature tensor of shape (batch_size, output_dim).
        '''
        flattened_input = input_tensor.view(-1, self.flatten_dim)  # Flatten the input tensor
        hidden_features1 = F.relu(self.fc_layer1(flattened_input))  # First hidden layer
        hidden_features2 = F.relu(self.fc_layer2(hidden_features1)) # Second hidden layer
        output_features = self.output_layer(hidden_features2)       # Output layer
        return output_features

class BarlowTwins(nn.Module):
    '''
    Barlow Twins architecture for self-supervised learning.

    Parameters
    ----------
    in_features : int, optional
        Number of input features, default is 256 * 16 * 16 * 16.
    proj_channels : int, optional
        Number of channels in the projector layers, default is 512.
    '''
    def __init__(self, in_features=256 * 16 * 16 * 16, proj_channels=512):
        super(BarlowTwins, self).__init__()

        # Define the projector network
        projector_layers = []
        for layer_index in range(3):  # Three layers for the projector
            if layer_index == 0:
                projector_layers.append(nn.Linear(in_features, proj_channels, bias=False))  # Input to first hidden layer
            else:
                projector_layers.append(nn.Linear(proj_channels, proj_channels, bias=False))  # Hidden to hidden/output layers

            if layer_index < 2:  # Add BatchNorm and ReLU to the first two layers
                projector_layers.append(nn.BatchNorm1d(proj_channels))
                projector_layers.append(nn.ReLU(inplace=True))

        self.projector = nn.Sequential(*projector_layers)

    def forward(self, input_view1, input_view2):
        '''
        Forward pass for the Barlow Twins network.

        Parameters
        ----------
        input_view1 : torch.Tensor
            First view of the input tensor of shape (batch_size, 256, 16, 16, 16).
        input_view2 : torch.Tensor
            Second view of the input tensor of shape (batch_size, 256, 16, 16, 16).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Projected representations for the two input views.
        '''
        flattened_view1 = input_view1.view(-1, 256 * 16 * 16 * 16)  # Flatten the first view
        flattened_view2 = input_view2.view(-1, 256 * 16 * 16 * 16)  # Flatten the second view

        projection1 = self.projector(flattened_view1)  # Project the first view
        projection2 = self.projector(flattened_view2)  # Project the second view

        return projection1, projection2
