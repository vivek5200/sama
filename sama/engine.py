"""Sama v0.3 — specialist dispatch engine."""

import json
import os
import time
from typing import Dict, Union
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from .encoder import FrozenEncoder
from .format import format_question, FORMAT_VERSION, check_format_compatibility, FormatVersionError
from .heads import load_head, HEAD_CLASSES
from .calibrate import apply_power_scaling
from .types import Decision, DecisionResponse


class DecisionEngine:
    """Dispatch wrapper: routes questions to specialist heads by type."""

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = model_dir
        self.device_arg = device

        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        # Validate format version
        cfg_format = self.config.get("format_version", "1.0.0")
        check_format_compatibility(cfg_format)

        # Load encoder
        encoder_id = self.config["encoder"]
        self.encoder = FrozenEncoder(encoder_id, device=device)
        hidden_dim = self.config.get("hidden_dim", self.encoder.hidden_dim)

        # Load all available heads
        self.heads: Dict[str, torch.nn.Module] = {}
        self.calibrations: Dict[str, dict] = {}

        for head_type, meta in self.config.get("heads", {}).items():
            # Validate head format compatibility
            head_format = meta.get("format_version", "1.0.0")
            check_format_compatibility(head_format)

            head_path = os.path.join(model_dir, meta["file"])
            if not os.path.exists(head_path):
                continue
            self.heads[head_type] = load_head(head_type, hidden_dim, head_path, self.encoder.device)
            self.calibrations[head_type] = meta.get("calibration", {})

        # --- Backward compatibility: v0.1 / v0.2 single-head format ---
        # If config.json declares "head_architecture" (old style), auto-detect and register
        # as a "choice" head.
        if not self.heads and "head_architecture" in self.config:
            self._load_legacy_head(model_dir, hidden_dim)

    def _load_legacy_head(self, model_dir: str, hidden_dim: int):
        """Load a v0.1/v0.2 style head (single choice head)."""
        from .heads import ChoiceHead

        head_path = os.path.join(model_dir, "head.pt")
        state = torch.load(head_path, map_location="cpu")

        # Rename old keys
        renamed = {}
        for k, v in state.items():
            if k.startswith("choice_head."):
                renamed[k.replace("choice_head.", "head.")] = v
            else:
                renamed[k] = v

        head = ChoiceHead(hidden_dim=hidden_dim)
        head.load_state_dict(renamed)
        head = head.to(self.encoder.device).eval()
        for p in head.parameters():
            p.requires_grad = False

        self.heads = {"choice": head}
        T = self.config.get("calibration", {}).get("T", 1.0)
        self.calibrations = {"choice": {"temperature": T, "method": "power_scaling"}}

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, device: str = "auto", cache_dir=None):
        if os.path.isdir(repo_id_or_path):
            return cls(repo_id_or_path, device=device)
        model_dir = snapshot_download(
            repo_id_or_path, cache_dir=cache_dir,
            allow_patterns=["*.pt", "*.json"],
        )
        return cls(model_dir, device=device)

    @torch.no_grad()
    def decide(self, state: str, questions: dict,
               auto_threshold: float = 0.9,
               review_threshold: float = 0.7) -> DecisionResponse:
        t_total = time.time()

        # 1. Format all questions
        qids = list(questions.keys())
        texts = [format_question(state, questions[qid]) for qid in qids]

        # 2. Encode once
        t_enc = time.time()
        embeddings = self.encoder.encode_last_token(texts)
        encode_ms = (time.time() - t_enc) * 1000

        # 3. Group by head type
        by_type = {}
        for i, qid in enumerate(qids):
            t = questions[qid]["type"]
            by_type.setdefault(t, []).append(i)

        # 4. Check availability
        for t in by_type:
            if t not in self.heads:
                raise MissingHeadError(
                    f"No {t!r} head loaded. Available: {list(self.heads.keys())}. "
                    f"For multi-task or choice-only heads, load a different model."
                )

        # 5. Run each head on its batch
        decisions: Dict[str, Decision] = {}
        t_head = time.time()

        for head_type, indices in by_type.items():
            head = self.heads[head_type]
            h = embeddings[indices]
            logits = head(h)
            T = self.calibrations[head_type].get("temperature", 1.0)

            for j, i in enumerate(indices):
                qid = qids[i]
                q = questions[qid]

                if head_type == "choice":
                    k = len(q["options"])
                     # Need to index properly into logits by adding dim 0 if multiple questions
                    # Wait, if h is (batch, hidden), logits is (batch, options)
                    probs = F.softmax(logits[j, :k], dim=-1).cpu().numpy()
                    # Apply power scaling on the max probability
                    pred = int(probs.argmax())
                    raw_conf = float(probs.max())
                    cal_conf = apply_power_scaling(raw_conf, T)
                    decisions[qid] = Decision(
                        answer=q["options"][pred] if isinstance(q["options"], list) else list(q["options"].keys())[pred],
                        probabilities=[float(p) for p in probs],
                        confidence=cal_conf,
                        action=self._action(cal_conf, auto_threshold, review_threshold),
                        status="confident" if cal_conf >= review_threshold else "uncertain",
                    )

                elif head_type == "noul":
                    p_yes = torch.sigmoid(logits[j]).item()
                    conf = max(p_yes, 1 - p_yes)
                    cal_conf = apply_power_scaling(conf, T)
                    decisions[qid] = Decision(
                        answer=bool(p_yes > 0.5),
                        probabilities=[float(1 - p_yes), float(p_yes)],
                        confidence=cal_conf,
                        action=self._action(cal_conf, auto_threshold, review_threshold),
                        status="confident" if cal_conf >= review_threshold else "uncertain",
                    )

                elif head_type == "score":
                    k = len(q["levels"])
                    probs = F.softmax(logits[j, :k], dim=-1).cpu().numpy()
                    pred = int(probs.argmax())
                    raw_conf = float(probs.max())
                    cal_conf = apply_power_scaling(raw_conf, T)
                    decisions[qid] = Decision(
                        answer=q["levels"][pred],
                        probabilities=[float(p) for p in probs],
                        confidence=cal_conf,
                        action=self._action(cal_conf, auto_threshold, review_threshold),
                        status="confident" if cal_conf >= review_threshold else "uncertain",
                    )

        head_ms = (time.time() - t_head) * 1000
        total_ms = (time.time() - t_total) * 1000

        return DecisionResponse(
            decisions=decisions,
            meta={
                "total_latency_ms": total_ms,
                "encode_latency_ms": encode_ms,
                "head_latency_ms": head_ms,
                "encoder_id": self.config.get("encoder"),
                "format_version": FORMAT_VERSION,
                "package_version": self.config.get("package_version", "0.3.0"),
            },
        )

    @staticmethod
    def _action(conf: float, auto_t: float, review_t: float) -> str:
        if conf >= auto_t:
            return "auto"
        if conf >= review_t:
            return "soft_review"
        return "escalate"


class MissingHeadError(RuntimeError):
    pass