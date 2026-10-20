"""Frozen encoder loading and forward pass."""

import torch
from transformers import AutoModel, AutoTokenizer


class FrozenEncoder:
    def __init__(self, encoder_id: str, device: str = "auto"):
        self.encoder_id = encoder_id
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModel.from_pretrained(
                encoder_id, dtype=torch.float16, device_map=device,
            )
        except (ValueError, ImportError):
            # Fallback if accelerate not installed
            target = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModel.from_pretrained(
                encoder_id, dtype=torch.float16,
            ).to(target)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size

    @property
    def device(self):
        return self.model.device

    @torch.no_grad()
    def encode_last_token(self, texts, max_length: int = 320):
        """Encode a list of strings, return (N, hidden_dim) last-token hidden states."""
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length).to(self.device)
        out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1].float()
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0]), seq_lens]
        return pooled
