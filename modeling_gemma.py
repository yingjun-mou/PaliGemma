"""This module corresponds to the Gemma language model portion in Figure 1 of the PaliGemma paper. Source: https://arxiv.org/abs/2407.07726."""

from typing import Optional, Tuple
from modeling_siglip import SiglipVisionModel
import torch


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