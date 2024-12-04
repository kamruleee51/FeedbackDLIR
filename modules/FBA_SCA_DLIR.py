import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.SCA_RegUNet import *
from modules.layers import *

# Define the spatiotemporal attention mechanism for multi-frame feature correlation and refinement
class SpatiotemporalAttention(nn.Module):
    """
    A class implementing spatiotemporal attention for multi-frame feature correlation and refinement.
    """

    def __init__(self, channel_count, spatial_dim=60 * 60):
        """
        Initialize the SpatiotemporalAttention module.

        Parameters
        ----------
        channel_count : int
            Number of channels in the input features.
        spatial_dim : int, optional
            Flattened spatial dimensions (default is 60 * 60).
        """
        super(SpatiotemporalAttention, self).__init__()
        # Linear transformation for exemplar features
        self.exemplar_transform = nn.Linear(channel_count, channel_count, bias=False)
        # Linear transformation for query features
        self.query_transform = nn.Linear(channel_count, channel_count, bias=False)
        self.channel_count = channel_count
        self.spatial_dim = spatial_dim
        # Convolution layer for generating feature gate
        self.feature_gate = nn.Conv3d(256, 1, kernel_size=1, bias=False)
        # Sigmoid activation for attention masks
        self.activation = nn.Sigmoid()

    def forward(self, exemplar_features, query_features):
        """
        Forward pass for spatiotemporal attention.

        Parameters
        ----------
        exemplar_features : torch.Tensor
            Tensor representing the exemplar features of shape (batch_size, channels, depth, height, width).
        query_features : torch.Tensor
            Tensor representing the query features of shape (batch_size, channels, depth, height, width).

        Returns
        -------
        tuple
            Tuple containing:
            - Exemplar attention mask.
            - Query attention mask.
            - Correlation matrix from exemplar to query.
            - Correlation matrix from query to exemplar.
        """
        # Extract spatial dimensions
        feature_size = query_features.size()[2:]
        flattened_dim = feature_size[0] * feature_size[1] * feature_size[2]

        # Flatten spatial dimensions of exemplar and query features
        exemplar_flat = exemplar_features.view(-1, query_features.size()[1], flattened_dim)
        query_flat = query_features.view(-1, query_features.size()[1], flattened_dim)

        # Transpose feature tensors for correlation calculation
        exemplar_flat_transposed = torch.transpose(exemplar_flat, 1, 2).contiguous()
        query_flat_transposed = torch.transpose(query_flat, 1, 2).contiguous()

        # Apply linear transformations
        exemplar_corr = self.exemplar_transform(exemplar_flat_transposed)
        query_corr = self.query_transform(query_flat_transposed)

        # Compute correlation matrices
        correlation_exemplar_to_query = torch.bmm(exemplar_corr, query_flat)
        correlation_query_to_exemplar = torch.bmm(query_corr, exemplar_flat)

        # Apply softmax to obtain attention weights
        attention_weights_exemplar_to_query = F.softmax(
            torch.transpose(correlation_exemplar_to_query, 1, 2), dim=1
        )
        attention_weights_query_to_exemplar = F.softmax(
            torch.transpose(correlation_query_to_exemplar, 1, 2), dim=1
        )

        # Apply attention weights to features
        exemplar_attention = torch.bmm(exemplar_flat, attention_weights_exemplar_to_query).contiguous()
        query_attention = torch.bmm(query_flat, attention_weights_query_to_exemplar).contiguous()

        # Reshape attention outputs to match original feature dimensions
        exemplar_attention_reshaped = exemplar_attention.view(
            -1, query_features.size()[1], feature_size[0], feature_size[1], feature_size[2]
        )
        query_attention_reshaped = query_attention.view(
            -1, query_features.size()[1], feature_size[0], feature_size[1], feature_size[2]
        )

        # Generate attention masks using feature gate
        exemplar_attention_mask = self.feature_gate(exemplar_attention_reshaped)
        query_attention_mask = self.feature_gate(query_attention_reshaped)

        # Apply sigmoid activation to masks
        exemplar_attention_mask = self.activation(exemplar_attention_mask)
        query_attention_mask = self.activation(query_attention_mask)

        # Return attention masks and correlation matrices
        return exemplar_attention_mask, query_attention_mask, attention_weights_exemplar_to_query, attention_weights_query_to_exemplar


# Define the model for feature extraction and registration
class FBA_SCA_DLIR3D(nn.Module):
    """
    A model that integrates coattention-based feature extraction and spatial transformation for 
    feature matching and registration tasks. It uses attention mechanisms and a 3D U-Net architecture.
    """
    
    def __init__(self, device, channel_count=256, spatial_dim=60*60):
        super(FBA_SCA_DLIR3D, self).__init__()
        
        # Encoder module for feature extraction
        self.coattention_encoder = encoder(channel_count).to(device)
        
        # 3D U-Net for feature transformation and registration prediction
        self.unet = SCARegUNet3D()
        
        # BarlowTwins module for self-supervised learning
        self.barlow_twins = BarlowTwins()
        
        # Spatial transformer for image warping
        self.spatial_transform = SpatialTransformer()
        
        # Activation functions
        self.softmax = nn.Sigmoid()
        self.prelu = nn.ReLU(inplace=True)
        
        # Convolution layer for combining attention
        self.att_combiner = nn.Conv3d(1, 1, kernel_size=3, stride=1)
        
        # Spatiotemporal attention mechanism
        self.attention = SpatiotemporalAttention(channel_count)

        # Initialize weights for Conv3d and GroupNorm layers
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights for Conv3d and GroupNorm layers with specific strategies.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fixed_image, moving_image):
        """
        Forward pass for the FBA_SCA_DLIR3D model.

        Parameters
        ----------
        fixed_image : torch.Tensor
            Fixed image in the registration task.
        moving_image : torch.Tensor
            Moving image in the registration task.

        Returns
        -------
        tuple
            Outputs including predicted displacement, attention-applied images, and learned representations.
        """
        # Extract features from fixed and moving images
        fixed_features = self.coattention_encoder(fixed_image)
        moving_features = self.coattention_encoder(moving_image)

        # Compute attention maps between fixed and moving features
        fixed_attention, moving_attention, _, _ = self.attention(fixed_features, moving_features)
        
        # Combine attention maps with moving and fixed images
        moving_attention_mask = self.att_combiner(moving_attention)
        fixed_attention_mask = self.att_combiner(fixed_attention)

        # Upsample attention masks to match image dimensions
        moving_attention_mask = F.upsample(moving_attention_mask, fixed_image.size()[2:], mode='trilinear')
        fixed_attention_mask = F.upsample(fixed_attention_mask, fixed_image.size()[2:], mode='trilinear')

        # Apply attention masks to images
        fixed_image_with_attention = fixed_image * fixed_attention_mask
        moving_image_with_attention = moving_image * moving_attention_mask

        # Use UNet to predict registration displacements
        registration_prediction = self.unet(torch.cat([fixed_image_with_attention, moving_image_with_attention], dim=1))

        # Warp the fixed image using the predicted displacement
        warped_fixed_image = self.spatial_transform(moving_image, registration_prediction)
        
        # Extract features from the warped fixed image
        warped_fixed_features = self.coattention_encoder(warped_fixed_image)
        
        # Compute attention between warped fixed and fixed features
        fixed_attention_warped, warped_fixed_attention, B1, B2 = self.attention(fixed_features, warped_fixed_features)
        # B1= attention_weights_exemplar_to_query
        # B2= attention_weights_query_to_exemplar 
        
        # Combine warped fixed image with attention masks
        warped_fixed_attention_mask = self.att_combiner(warped_fixed_attention)
        warped_fixed_attention_mask = F.upsample(warped_fixed_attention_mask, fixed_image.size()[2:], mode='trilinear')

        warped_fixed_image_with_attention = warped_fixed_image * warped_fixed_attention_mask

        # Predict final registration output
        final_registration_prediction = self.unet(torch.cat([warped_fixed_image_with_attention, moving_image_with_attention], dim=1))

        # Compute self-supervised BarlowTwins projections
        z1, z2 = self.barlow_twins(fixed_features, moving_features)
        # z1 = projection1
        # z2 = projection2

        # Return outputs including predictions, attention-applied images, and feature representations
        return final_registration_prediction, fixed_image_with_attention, moving_image_with_attention, fixed_attention_mask, moving_attention_mask, z1, z2, B1, B2
