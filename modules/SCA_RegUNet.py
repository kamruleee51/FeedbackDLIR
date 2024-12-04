import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import *  # Importing custom-defined layers

# Attention-Based 3D U-Net for Registration
class SCARegUNet3D(nn.Module):
    '''
    Attention U-Net for volumetric image registration with channel and spatial attention mechanisms.
    '''
    def __init__(self):
        super(SCARegUNet3D, self).__init__()
        
        # Define the number of filters for each level in the U-Net
        filters_per_level = [2, 128, 256, 512, 64, 3]  # [input, enc-1, enc-2, enc-3, dec-1, output]

        # Encoder blocks with attention mechanisms
        self.encoder_block1 = ResBlock_Attention(filters_per_level[0], filters_per_level[1])
        self.encoder_block2 = ResBlock_Attention(filters_per_level[1], filters_per_level[2])
        self.encoder_block3 = ResBlock_Attention(filters_per_level[2], filters_per_level[3])

        # Bottleneck block at the deepest level of the U-Net
        self.bottleneck_block = ResBlock_Attention(filters_per_level[3], filters_per_level[3])

        # Decoder blocks with skip connections
        self.decoder_block3 = ResBlock_Attention(filters_per_level[3] + filters_per_level[3], filters_per_level[2])
        self.decoder_block2 = ResBlock_Attention(filters_per_level[2] + filters_per_level[2], filters_per_level[1])
        self.decoder_block1 = ResBlock_Attention(filters_per_level[1] + filters_per_level[1], filters_per_level[4])

        # Final convolution layer to produce the output volume
        self.final_conv = nn.Conv3d(filters_per_level[4], filters_per_level[5], kernel_size=3, padding=1)

        # Pooling layer for downsampling in the encoder path
        self.maxpool3d = nn.MaxPool3d(kernel_size=2)

        # Upsampling layer for upsampling in the decoder path
        self.upsample3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input_volume):
        '''
        Forward pass through the Attention_RegUNet3D model.

        Parameters
        ----------
        input_volume : torch.Tensor
            Input tensor with shape (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after the forward pass.
        '''
        # Encoder Path
        encoder_features1 = self.encoder_block1(input_volume)  # First encoder block
        pooled_features1 = self.maxpool3d(encoder_features1)  # Downsample features

        encoder_features2 = self.encoder_block2(pooled_features1)  # Second encoder block
        pooled_features2 = self.maxpool3d(encoder_features2)  # Downsample further

        encoder_features3 = self.encoder_block3(pooled_features2)  # Third encoder block
        pooled_features3 = self.maxpool3d(encoder_features3)  # Downsample to bottleneck

        # Bottleneck
        bottleneck_features = self.bottleneck_block(pooled_features3)  # Bottleneck features

        # Decoder Path
        upsampled_features3 = self.upsample3d(bottleneck_features)  # Upsample bottleneck features
        concatenated_features3 = torch.cat([upsampled_features3, encoder_features3], dim=1)  # Skip connection
        decoder_features3 = self.decoder_block3(concatenated_features3)  # Third decoder block

        upsampled_features2 = self.upsample3d(decoder_features3)  # Upsample
        concatenated_features2 = torch.cat([upsampled_features2, encoder_features2], dim=1)  # Skip connection
        decoder_features2 = self.decoder_block2(concatenated_features2)  # Second decoder block

        upsampled_features1 = self.upsample3d(decoder_features2)  # Upsample
        concatenated_features1 = torch.cat([upsampled_features1, encoder_features1], dim=1)  # Skip connection
        decoder_features1 = self.decoder_block1(concatenated_features1)  # First decoder block

        # Final Convolution
        output_volume = self.final_conv(decoder_features1)  # Generate the output volume
        
        return output_volume
