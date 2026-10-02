import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import Dict
from .types import Decision, DecisionResponse


class TypedDecisionHead(nn.Module):
    def __init__(self, hidden_dim=896, proj_dim=512, max_options=8, dropout=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.choice_head = nn.Linear(proj_dim, max_options)

    def forward(self, h):
        return {"choice_logits": self.choice_head(self.proj(h))}


class DecisionEngine:
    def __init__(self, repo_id, device="auto", temperature=None):
        self.repo_id = repo_id
        self.device = device

        # Load head + config from local path or HuggingFace
        if os.path.isdir(repo_id):
            head_path = os.path.join(repo_id, "head.pt")
            config_path = os.path.join(repo_id, "config.json")
        else:
            head_path = hf_hub_download(repo_id=repo_id, filename="head.pt")
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        # Load encoder
        encoder_id = self.config.get("encoder", "Qwen/Qwen2.5-0.5B")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.encoder = AutoModel.from_pretrained(
            encoder_id, dtype=torch.bfloat16, device_map=device,
            )
        except (ValueError, ImportError):
            # Fallback: no accelerate, load to a specific device
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = AutoModel.from_pretrained(
            encoder_id, dtype=torch.bfloat16,
            ).to(target_device)
        self.encoder.eval()

        # Load head
        hidden_dim = self.config.get("hidden_dim", 896)
        self.head = TypedDecisionHead(hidden_dim=hidden_dim)
        self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        self.head = self.head.to(self.encoder.device)
        self.head.eval()

        # Temperature from config
        self.temperature = temperature or self.config.get("calibration", {}).get("T", 1.0)

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        return cls(repo_id, **kwargs)

    def _format_state(self, state, questions):
        """Match the training format exactly."""
        parts = [state, ""]
        for qid, q in questions.items():
            parts.append(f"Question: {q['instructions']}")
            if q.get("type") == "choice":
                opts = q["options"]
                if isinstance(opts, dict):
                    opts = list(opts.keys())
                # Match training: space-separated (A) x (B) y
                opts_str = " ".join(f"({chr(65+j)}) {o}" for j, o in enumerate(opts))
                parts.append(f"Options: {opts_str}")
        parts.append("Answer:")
        return "\n".join(parts)
    @torch.no_grad()
    def decide(self, state, questions, auto_threshold=0.9, review_threshold=0.7):
        text = self._format_state(state, questions)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=2048).to(self.encoder.device)
        out = self.encoder(**enc, output_hidden_states=True)
        h = out.hidden_states[-1].float()
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0]), seq_lens]

        head_out = self.head(pooled)

        decisions = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                k = len(q["options"]) if isinstance(q["options"], list) else len(q["options"])
                logits = head_out["choice_logits"][0, :k] / self.temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                pred = int(probs.argmax())
                conf = float(probs.max())
                opts = q["options"]
                answer = opts[pred] if isinstance(opts, list) else list(opts.keys())[pred]
                action = "auto" if conf >= auto_threshold else (
                    "soft_review" if conf >= review_threshold else "escalate")
                decisions[qid] = Decision(
                    answer=answer,
                    probabilities=[float(p) for p in probs],
                    confidence=conf,
                    action=action,
                    status="confident" if conf >= review_threshold else "uncertain",
                )

        return DecisionResponse(decisions=decisions)