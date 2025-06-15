from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768, # size of the embedding vector
            intermediate_size=3072, # size of the linear layer used for forward
            num_hidden_layers=12, # number of layers for vision transformer
            num_attention_heads=12, # number of attention heads in the multi-head attention
            num_channels=3, # number of channels each image has, equals 3 for R,G,B
            image_size=224, # this depends on the input image size for PaliGemma
            patch_size=16, # dive image into patches of 16x16
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None, # number of image embeddings for each image
            **kwargs   
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        # While an image encoder transform an image to an embedding, a vision transformer
        # can convert an image into multiple encodings.
        self.num_image_tokens = num_image_tokens