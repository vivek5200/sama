import os
import tempfile

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. Load Encoder and Base Head
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
# Qwen's tokenizer may not define a padding token; use EOS for padded batches.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

encoder = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", dtype=torch.float32
)
encoder.config.pad_token_id = tokenizer.pad_token_id
encoder.eval()


class ChoiceHead5(nn.Module):
    def __init__(self, hd=1536, pd=512, mo=5, dr=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hd, pd),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(pd, pd),
            nn.GELU(),
            nn.Dropout(dr),
        )
        self.head = nn.Linear(pd, mo)

    def forward(self, h):
        return self.head(self.proj(h))


head = ChoiceHead5()
head.load_state_dict(torch.load("choice_head_5class.pt", map_location="cpu"))
head.eval()

OPTIONS = ["Technical", "Financial", "General", "Sales", "Internal"]


# 2. Inference Function
def route_ticket(ticket_text):
    if not ticket_text.strip():
        return "Please enter a ticket."

    text = (
        f"{ticket_text}\n\nQuestion: Which support queue should handle this ticket?\n"
        "Type: choice\nOptions: "
        + " ".join(f"({chr(65 + j)}) {opt}" for j, opt in enumerate(OPTIONS))
        + "\nAnswer:"
    )

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320)
    with torch.no_grad():
        out = encoder(**enc, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1].float()
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0]), seq_lens]

        logits = head(pooled)[0]
        probs = F.softmax(logits / 1.071, dim=-1).numpy()  # T=1.071

        best_idx = probs.argmax()
        best_cluster = OPTIONS[best_idx]
        conf = probs[best_idx]

        action = "✅ AUTO-ROUTE" if conf >= 0.80 else "⚠️ ESCALATE TO HUMAN/GPT-4"

        return (
            f"Predicted Queue: {best_cluster}\nConfidence: {conf:.1%}\n"
            f"Action: {action}"
        )


# 3. Training Function
def train_custom_head(file):
    if file is None:
        return None, None, "Please upload a CSV file."

    try:
        df = pd.read_csv(file)
        if "text" not in df.columns or "label" not in df.columns:
            return None, None, "Error: CSV must have 'text' and 'label' columns."
        if len(df) == 0:
            return None, None, "Error: CSV must contain at least one row."
        if len(df) > 300:
            return (
                None,
                None,
                "Error: Max 300 rows allowed in the web demo (CPU limits). "
                "Use the GitHub script for larger datasets.",
            )
        if df[["text", "label"]].isna().any().any():
            return None, None, "Error: 'text' and 'label' values cannot be empty."

        texts = df["text"].astype(str).str.strip().tolist()
        labels_list = df["label"].astype(str).str.strip().tolist()
        if any(not text for text in texts):
            return None, None, "Error: 'text' values cannot be empty."
        if any(not label for label in labels_list):
            return None, None, "Error: 'label' values cannot be empty."

        # Keep the class order stable in the order labels first appear in the CSV.
        labels_list = list(dict.fromkeys(labels_list))
        if len(labels_list) < 2:
            return None, None, "Error: CSV must contain at least two distinct labels."
        label2id = {label: i for i, label in enumerate(labels_list)}
        labels = torch.tensor(
            [label2id[label] for label in df["label"].astype(str).str.strip()],
            dtype=torch.long,
        )
        num_classes = len(labels_list)

        # Encode with the frozen encoder in small batches to limit CPU memory use.
        embeddings = []
        for i in range(0, len(texts), 8):
            batch = texts[i : i + 8]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320,
            )
            with torch.no_grad():
                out = encoder(**enc, output_hidden_states=True, logits_to_keep=1)
                h = out.hidden_states[-1].float()
                seq_lens = enc["attention_mask"].sum(dim=1) - 1
                pooled = h[torch.arange(h.shape[0]), seq_lens]
                embeddings.append(pooled.cpu())
        X = torch.cat(embeddings, dim=0)

        # Train a compact classifier on the frozen encoder features.
        custom_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes),
        )
        opt = torch.optim.AdamW(custom_head.parameters(), lr=2e-5)
        final_loss = 0.0

        for _ in range(30):
            custom_head.train()
            opt.zero_grad()
            logits = custom_head(X)
            ce = F.cross_entropy(logits, labels)
            p = F.softmax(logits, dim=-1)
            oh = F.one_hot(labels, num_classes).float()
            brier = ((p - oh) ** 2).sum(-1).mean()
            loss = ce + 0.5 * brier
            loss.backward()
            opt.step()
            final_loss = loss.item()

        custom_head.eval()

        # Keep the files in a temporary directory so Gradio can serve them for download.
        temp_dir = tempfile.mkdtemp(prefix="sama-custom-head-")
        model_path = os.path.join(temp_dir, "custom_head.pt")
        torch.save(custom_head.state_dict(), model_path)

        mapping_path = os.path.join(temp_dir, "label_mapping.txt")
        with open(mapping_path, "w", encoding="utf-8") as mapping_file:
            for i, label in enumerate(labels_list):
                mapping_file.write(f"{i}: {label}\n")

        msg = (
            f"✅ Successfully trained on {len(texts)} examples for "
            f"{num_classes} classes (30 epochs).\nFinal RLCD loss: {final_loss:.4f}\n\n"
            "Label Mapping:\n"
            + "\n".join(f"{i}: {label}" for i, label in enumerate(labels_list))
        )
        return model_path, mapping_path, msg

    except Exception as e:
        return None, None, f"Error: {e}"


# 4. Build the UI using gr.Blocks
with gr.Blocks(title="Sama ITSM Router") as demo:
    gr.Markdown(
        "# Sama ITSM Router\n"
        "Route a ticket with the pretrained model, or train a custom classification "
        "head using a small labeled CSV dataset."
    )

    with gr.Tab("Inference Demo"):
        ticket_input = gr.Textbox(lines=5, label="Paste Ticket Here")
        route_button = gr.Button("Submit", variant="primary")
        route_output = gr.Textbox(label="Routing Decision")
        route_button.click(
            fn=route_ticket, inputs=ticket_input, outputs=route_output
        )

    with gr.Tab("Train Your Own"):
        gr.Markdown(
            "Upload a CSV with **text** and **label** columns (up to 300 rows). "
            "The encoder stays frozen; only a new classification head is trained."
        )
        csv_file = gr.File(
            label="Training CSV",
            file_types=[".csv"],
            type="filepath",
        )
        train_button = gr.Button("Train custom head", variant="primary")
        with gr.Row():
            model_download = gr.File(label="Download trained head (.pt)")
            mapping_download = gr.File(label="Download label mapping (.txt)")
        train_logs = gr.Textbox(label="Training log", lines=8, interactive=False)
        train_button.click(
            fn=train_custom_head,
            inputs=csv_file,
            outputs=[model_download, mapping_download, train_logs],
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
