"""This module corresponds to the Gemma language model portion in Figure 1 of the PaliGemma paper. Source: https://arxiv.org/abs/2407.07726."""

from typing import Optional, Tuple
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import torch


class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig():
    """The config passed into PaliGemma - the final model consists of both SigLIP and Gemma."""
    
    def __init__(
            self,
            vision_config=None, # passed into SigLIP
            text_config=None, # passed into Gemma
            ignore_index=100,
            image_token_index=25600, # the index for placholder token <image>
            vocab_size=257152,
            projection_dim=2048, # the dimension of image embedding came out of the linear layer
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)
        self.vision_config.projection_dim = projection_dim


class PaliGemmaForConditionalGeneration(nn.Module):
    """Conditions the generation of text on the input image."""
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        """Re-use layer parameters of the initial embedding layer (vocab_sie -> emb_size)
           in final linaer layer (emb_size -> vocab_size), since these two layers are doing the opossite tranformations.
           See the Transformer decoder architecture in the "Attention Is All You Need" paper.
           When the vocab size is large, this re-use can reduce the number of params by 10%.
        """
        return self.language_model.tie_weights()
    
    def forwrad(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCahe] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings
        # (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens 
        image_features, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embds, input_ids, attention_mask, kv_cache)