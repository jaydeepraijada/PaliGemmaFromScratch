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
            **kwargs #catch-all for any extra paramet`ers. Means we can add extra parameters without breaking the code
            
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


class SiglipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        self.scale = self.head_dim**-0.5 #scaling factor for dot product attention 
        self.dropout = config.attention_dropout

        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        #[Batch_size, Num_Patches, Embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) # [Batch_size, Num_Patches, Embed_dim]
        key_states = self.k_proj(hidden_states)   # [Batch_size, Num_Patches, Embed_dim]
        value_states = self.v_proj(hidden_states) # [Batch_size, Num_Patches, Embed_dim]

        # we are splitting the embed_dim into multiple heads
        # [Batch_size, Num_Patches, Embed_dim] -> [Batch_size, Num_Patches, Num_heads, Head_dim] -> [Batch_size, Num_heads, Num_Patches, Head_dim]  
        # We do this becasue multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions
        # we transpose so that we can perform the dot product attention for each head in parallel, by treating each head independently

        
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)  

        # Calculate attention scores

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_attention_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_attention_heads, seq_len, seq_len)}, but is {attn_weights.size()}"
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(query_states.dtype)
        # Apply dropout to attention weights
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply attention weights with value states to get context layer
        attn_output = torch.matmul(attn_weights, value_states) # [Batch_size, Num_heads, Num_Patches, Head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim) # [Batch_size, Num_Patches, Embed_dim] 
        attn_output = self.out_proj(attn_output) # [Batch_size, Num_Patches, Embed_dim]
        return attn_output, attn_weights


        

class SiglipMLP(nn.Module):


    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, _ = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states

        residual2 = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual2 + hidden_states

        return hidden_states

class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states
    
    
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
        _, _, height, width = pixel_values.shape # [Batch_size, Channels, Height, Width]
        # Convolve the "patch_size" kernal over the image, with no overlapping patches since stride = patch_size
        patch_embeds = self.patch_embeddings(pixel_values) 

        # [Batch_size, Embed_dim, Num_patches_height, Num_patches_width] -> [Batch_size, Embed_dim, Num_patches] 
        #where num_patches = Num_patches_height * Num_patches_width
        embeddings = patch_embeds.flatten(2) #flatten height and width dimensions

        # [Batch_size, Embed_dim, Num_patches] -> [Batch_size, Num_patches, Embed_dim]
        embeddings = embeddings.transpose(1, 2)

        # Add position embeddings to each patch. Each Position embedding is of size [Embed_dim]
        position_embeddings = self.position_embeddings(self.position_ids)

        #[Batch_size, Num_patches, Embed_dim]
        embeddings = embeddings + position_embeddings

        return embeddings


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoderLayer(config)
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
    
