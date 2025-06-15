"""This module corresponds to the Gemma language model portion in Figure 1 of the PaliGemma paper. Source: https://arxiv.org/abs/2407.07726."""

from modeling_siglip import SiglipVisionModel


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
