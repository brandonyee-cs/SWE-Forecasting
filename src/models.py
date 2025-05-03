import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for SWETransUNet
    
    This layer converts the input tensor into embedded patches using 3D convolution.
    """
    def __init__(self, patch_size=(2, 4, 4), emb_dim=384):
        """
        Initialize the Patch Embedding layer
        
        Args:
            patch_size (tuple): Size of patches (depth, height, width)
            emb_dim (int): Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.conv = nn.Conv3d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Forward pass for Patch Embedding
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, 4, 12, 64, 64, 3]
            
        Returns:
            Tensor: Embedded patches
        """
        # Extract shapes
        batch_size, sensors, time_steps, height, width, channels = x.shape
        
        # Reshape to combine batch and sensors dimensions
        x = x.reshape(-1, time_steps, height, width, channels)
        
        # Permute to get channels first for Conv3D
        x = x.permute(0, 4, 1, 2, 3)  # [batch_size*sensors, 3, 12, 64, 64]
        
        # Apply 3D convolution
        x = self.conv(x)  # [batch_size*sensors, emb_dim, depth', height', width']
        
        # Get the new dimensions
        new_time, new_height, new_width = x.shape[2:]
        
        # Reshape and permute back
        x = x.reshape(batch_size, sensors, self.emb_dim, new_time, new_height, new_width)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [batch_size, sensors, depth', height', width', emb_dim]
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder layer for SWETransUNet
    
    This layer applies self-attention and feed-forward transformations.
    """
    def __init__(self, emb_dim=384, num_heads=4, hidden_dim=512, dropout=0.1):
        """
        Initialize the Transformer Encoder layer
        
        Args:
            emb_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension for feed-forward network
            dropout (float): Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for Transformer Encoder
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Transformed tensor
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward network with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell
    
    This cell combines convolutional operations with LSTM gating mechanisms
    for spatial-temporal data processing.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        """
        Initialize ConvLSTM Cell
        
        Args:
            input_dim (int): Number of input channels
            hidden_dim (int): Number of hidden channels
            kernel_size (int): Size of the convolutional kernel
            bias (bool): Whether to use bias in convolution
        """
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Single convolution for all gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 4 gates: input, forget, output, cell
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for ConvLSTM Cell
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, height, width]
            h_prev (Tensor): Previous hidden state
            c_prev (Tensor): Previous cell state
            
        Returns:
            tuple: (h_next, c_next) Next hidden and cell states
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Apply convolution
        conv_output = self.conv(combined)
        
        # Split the output into the four gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        
        # Apply activations to gates
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # cell input
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM layer
    
    Processes a sequence of tensors using ConvLSTMCell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, 
                 batch_first=True, bias=True, return_sequences=True):
        """
        Initialize ConvLSTM
        
        Args:
            input_dim (int): Number of input channels
            hidden_dim (int): Number of hidden channels
            kernel_size (int): Size of the convolutional kernel
            num_layers (int): Number of ConvLSTM layers stacked on each other
            batch_first (bool): If True, input is (batch, time, channels, height, width)
            bias (bool): Whether to use bias in convolution
            return_sequences (bool): If True, return the entire sequence, otherwise just the last output
        """
        super(ConvLSTM, self).__init__()
        
        # Save parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_sequences = return_sequences
        
        # Create a list of ConvLSTM cells
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias))
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass for ConvLSTM
        
        Args:
            x (Tensor): Input tensor of shape [batch, time, channels, height, width]
            hidden_state (tuple, optional): Initial hidden state and cell state
            
        Returns:
            Tensor: Output tensor
        """
        # Get batch and spatial sizes
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width)
        
        h, c = hidden_state
        
        # Container for output sequence
        output_sequence = []
        
        # Process the sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]  # Current time step
            
            # Process through layers
            for layer_idx in range(self.num_layers):
                # Input to this layer is either the input data (for first layer)
                # or the output from previous layer
                if layer_idx == 0:
                    layer_input = x_t
                else:
                    layer_input = h[layer_idx - 1]
                
                # Process through the ConvLSTM cell
                h[layer_idx], c[layer_idx] = self.cell_list[layer_idx](
                    layer_input, h[layer_idx], c[layer_idx]
                )
            
            # Store the output of the last layer
            if self.return_sequences:
                output_sequence.append(h[-1].unsqueeze(1))
        
        # Return output
        if self.return_sequences:
            # Concatenate all outputs if returning sequences
            return torch.cat(output_sequence, dim=1)
        else:
            # Return only the final output
            return h[-1]
    
    def _init_hidden(self, batch_size, height, width):
        """Initialize hidden and cell states"""
        # Initialize hidden state and cell state for each layer
        h = []
        c = []
        for i in range(self.num_layers):
            h.append(torch.zeros(batch_size, self.hidden_dim, height, width, 
                                device=self._get_device()))
            c.append(torch.zeros(batch_size, self.hidden_dim, height, width,
                                device=self._get_device()))
        return h, c
    
    def _get_device(self):
        """Get the device of the first parameter of the model"""
        return next(self.parameters()).device


class UNetDecoderBlock(nn.Module):
    """
    U-Net Decoder Block
    
    Performs upsampling and convolution operations for the decoder part of U-Net.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize U-Net Decoder Block
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(UNetDecoderBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass for U-Net Decoder Block
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        return self.conv(x)


class SWETransUNet(nn.Module):
    """
    Snow Water Equivalent Transformer U-Net Model
    
    This model combines Transformer, ConvLSTM, and U-Net architectures for 
    spatio-temporal data forecasting.
    """
    def __init__(self, in_channels=3, emb_dim=384, num_heads=4, hidden_dim=512, 
                 num_transformer_layers=2, patch_size=(2, 4, 4), dropout=0.1):
        """
        Initialize SWETransUNet
        
        Args:
            in_channels (int): Number of input channels (RGB = 3)
            emb_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension for feed-forward network
            num_transformer_layers (int): Number of transformer encoder layers
            patch_size (tuple): Size of patches (depth, height, width)
            dropout (float): Dropout rate
        """
        super(SWETransUNet, self).__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, emb_dim)
        
        # Transformer encoder layers
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(emb_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # ConvLSTM for temporal processing
        self.conv_lstm = ConvLSTM(
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            kernel_size=3,
            num_layers=1,
            batch_first=True,
            return_sequences=True
        )
        
        # U-Net decoder blocks with decreasing number of channels
        self.decoder_blocks = nn.ModuleList([
            UNetDecoderBlock(emb_dim, 192),
            UNetDecoderBlock(192, 96),
            UNetDecoderBlock(96, 48),
            UNetDecoderBlock(48, 32)
        ])
        
        # Final convolution to produce the output
        self.final_conv = nn.Conv2d(32, in_channels, kernel_size=1)
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        """
        Forward pass for SWETransUNet
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, 4, 12, 64, 64, 3]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, 64, 64, 3]
        """
        # Apply patch embedding
        x = self.patch_embed(x)  # [batch_size, 4, depth', height', width', emb_dim]
        
        # Get dimensions
        batch_size, sensors, depth, height, width, channels = x.shape
        
        # Reshape for transformer (flatten spatial dimensions)
        x = x.reshape(batch_size * sensors * depth, height * width, channels)
        
        # Apply transformer encoders
        for encoder in self.transformer_encoders:
            x = encoder(x)
        
        # Reshape for ConvLSTM
        x = x.reshape(batch_size * sensors, depth, height, width, channels)
        x = x.permute(0, 1, 4, 2, 3)  # [batch_size*sensors, depth, channels, height, width]
        
        # Apply ConvLSTM
        x = self.conv_lstm(x)  # [batch_size*sensors, depth, emb_dim, height, width]
        
        # Take the last time step
        x = x[:, -1]  # [batch_size*sensors, emb_dim, height, width]
        
        # Take only the first sensor's output (we're predicting from sensor 0)
        x = x[:batch_size]  # [batch_size, emb_dim, height, width]
        
        # Apply decoder blocks with upsampling
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            if i < 2:  # Upsample twice to reach 64x64
                x = self.upsample(x)
        
        # Final convolution
        x = self.final_conv(x)  # [batch_size, 3, 64, 64]
        
        # Permute to match output format
        x = x.permute(0, 2, 3, 1)  # [batch_size, 64, 64, 3]
        
        return x


# Alternative models for experimentation
class SWEUNet(nn.Module):
    """
    Simplified U-Net architecture for SWE forecasting without transformers
    
    This model is provided as an alternative to the main SWETransUNet model.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(SWEUNet, self).__init__()
        
        # Encoder
        self.encoder1 = self._make_encoder_block(in_channels, base_channels)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels*2)
        self.encoder3 = self._make_encoder_block(base_channels*2, base_channels*4)
        self.encoder4 = self._make_encoder_block(base_channels*4, base_channels*8)
        
        # Decoder
        self.decoder1 = self._make_decoder_block(base_channels*8, base_channels*4)
        self.decoder2 = self._make_decoder_block(base_channels*4, base_channels*2)
        self.decoder3 = self._make_decoder_block(base_channels*2, base_channels)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Reshape input for U-Net (taking only the most recent month and first sensor)
        # [batch_size, 4, 12, 64, 64, 3] -> [batch_size, 3, 64, 64]
        x = x[:, 0, -1].permute(0, 3, 1, 2)  # Select sensor 0, last time step, move channels first
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Decoder with skip connections
        dec1 = self.decoder1(self.upsample(enc4))
        dec2 = self.decoder2(self.upsample(dec1))
        dec3 = self.decoder3(self.upsample(dec2))
        
        # Final convolution
        output = self.final_conv(dec3)
        
        # Permute back to original format
        output = output.permute(0, 2, 3, 1)  # [batch_size, 64, 64, 3]
        
        return output


# Additional model component: Spatial Attention Module
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Computes attention weights based on spatial features.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # Compute channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Compute attention map
        attention = torch.sigmoid(self.conv(pooled))
        
        # Apply attention
        return x * attention


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    
    # Create a random input tensor
    x = torch.randn(batch_size, 4, 12, 64, 64, 3)
    
    # Initialize model
    model = SWETransUNet()
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test the alternative model
    alt_model = SWEUNet()
    alt_output = alt_model(x)
    print(f"Alternative model output shape: {alt_output.shape}")