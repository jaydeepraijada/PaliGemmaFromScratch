from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size: 768, #embedding dimension
            intermediate_size: 3072, #feedforward dimension
            num_hidden_layers: 12, #number of transformer layers    
            num_attention_heads: 12, #number of attention heads
            num_channels = 3, #number of input channels (RGB)
            image_size = 224, #input image size
            patch_size = 16, #patch size
            layer_norm_eps = 1e-6,#additional epsilon for layer norm, added to prevent division by zero
            attention_dropout = 0.0, #dropout rate for attention probabilities
            num_images_token: int = None, #number of image tokens to be used
            **kwargs #catch-all for any extra parameters. Means we can add extra parameters without breaking the code
            
    ):

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_images_token = num_images_token


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
    
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding = 'valid'
            )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_postions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_postions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_postions).expand((1, -1)), persistent=False
        )

    def forward(self, pixel_values):
        # pixel_values: [Batch_size, Channels, Height, Width]
        
        x = self.patch_embeddings(pixel_values)  # [Batch_size, Embed_dim, Num_patches_height, Num_patches_width]
        x = x.flatten(2)  # [Batch_size, Embed_dim, Num_patches]
        x = x.transpose(1, 2)  # [Batch_size, Num_patches, Embed_dim]

        x = x + self.position_embeddings  # Add position embeddings
        x = self.layernorm(x)

        return x


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        
        # pixel_values: [Batch_size, Channels, Height, Width] => [Bath_size, Num_patches, Embed_dim]

        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state
        
       
class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values):
        #[Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embed_dim]
        return self.vision_model(pixel_values = pixel_values)
    
