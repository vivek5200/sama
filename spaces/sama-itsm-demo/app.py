import ipaddress
import logging
import os
import tempfile
import threading
import time

# Gradio must know the managed cache directory before it is imported. Training
# artifacts are written there so Gradio can serve and later clean them up.
GRADIO_TEMP_DIR = os.path.abspath(
    os.environ.get("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "gradio"))
)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
os.makedirs(GRADIO_TEMP_DIR, mode=0o700, exist_ok=True)

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sama-itsm")

TRAIN_COOLDOWN_SECONDS = 5 * 60
MAX_RATE_LIMIT_KEYS = 10_000
_training_last_by_client = {}
_training_rate_lock = threading.Lock()


def _client_key(request):
    """Use Cloudflare's client address, falling back to the direct peer IP."""
    if request is not None:
        headers = request.headers or {}
        address = headers.get("cf-connecting-ip")
        if not address and request.client is not None:
            address = request.client.host
        if address:
            try:
                return str(ipaddress.ip_address(address.strip()))
            except ValueError:
                logger.warning("Ignoring invalid client IP supplied to rate limiter")
    # Fail closed: requests without a usable address share one cooldown bucket.
    return "unknown"


def _claim_training_slot(request):
    key = _client_key(request)
    now = time.monotonic()
    with _training_rate_lock:
        expired = [
            client
            for client, last_request in _training_last_by_client.items()
            if now - last_request >= TRAIN_COOLDOWN_SECONDS
        ]
        for client in expired:
            del _training_last_by_client[client]

        last_request = _training_last_by_client.get(key)
        if last_request is not None:
            remaining = TRAIN_COOLDOWN_SECONDS - (now - last_request)
            if remaining > 0:
                return False, max(1, int(remaining + 0.999))

        if len(_training_last_by_client) >= MAX_RATE_LIMIT_KEYS:
            return False, None

        _training_last_by_client[key] = now
    return True, 0


def _create_cache_file(suffix):
    fd, path = tempfile.mkstemp(
        prefix="sama-custom-head-", suffix=suffix, dir=GRADIO_TEMP_DIR
    )
    os.close(fd)
    return path


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
    if not ticket_text or not ticket_text.strip():
        return "⚠️ **Enter a ticket description to get a routing suggestion.**"

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
            f"### Suggested queue: {best_cluster}\n\n"
            f"**Confidence:** {conf:.1%}  \n"
            f"**Recommended action:** {action}\n\n"
            "This is a model suggestion. Review the ticket before taking action."
        )


# 3. Training Function
def train_custom_head(
    file,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    if file is None:
        return None, None, "⚠️ **Choose a CSV file first.**"

    try:
        file_path = file.name if hasattr(file, "name") else os.fspath(file)
    except (TypeError, ValueError):
        return None, None, "⚠️ **The uploaded file could not be read. Please upload it again.**"

    if os.path.splitext(file_path)[1].lower() != ".csv":
        return None, None, "⚠️ **This file does not have a `.csv` extension.**"

    try:
        if os.path.getsize(file_path) > 2 * 1024 * 1024:
            return None, None, "⚠️ **The CSV is larger than 2 MB.** Reduce its size and try again."
    except OSError:
        return None, None, "⚠️ **The uploaded file is no longer available. Please upload it again.**"

    try:
        progress(0.02, desc="Reading and checking your CSV")
        try:
            df = pd.read_csv(file_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            return None, None, "⚠️ **We couldn't read this as a valid CSV.** Check the file and try again."
        if "text" not in df.columns or "label" not in df.columns:
            return None, None, (
                "⚠️ **Your CSV needs columns named `text` and `label`.** "
                "Column names are case-sensitive."
            )
        if len(df) == 0:
            return None, None, "⚠️ **Your CSV has a header but no data rows.** Add at least two examples."
        if len(df) > 300:
            return (
                None,
                None,
                f"⚠️ **Your CSV has {len(df)} rows; the limit is 300.** "
                "Use the training script for a larger dataset.",
            )
        if df[["text", "label"]].isna().any().any():
            return None, None, "⚠️ **Some `text` or `label` cells are empty.** Fill them in and upload again."

        texts = df["text"].astype(str).str.strip().tolist()
        labels_list = df["label"].astype(str).str.strip().tolist()
        if any(not text for text in texts):
            return None, None, "⚠️ **Every row needs ticket text.** Remove blank `text` cells and try again."
        if any(not label for label in labels_list):
            return None, None, "⚠️ **Every row needs a label.** Remove blank `label` cells and try again."

        # Keep the class order stable in the order labels first appear in the CSV.
        labels_list = list(dict.fromkeys(labels_list))
        if len(labels_list) < 2:
            return None, None, "⚠️ **At least two different labels are required** to train a classifier."
        label2id = {label: i for i, label in enumerate(labels_list)}
        labels = torch.tensor(
            [label2id[label] for label in df["label"].astype(str).str.strip()],
            dtype=torch.long,
        )
        num_classes = len(labels_list)

        # Apply the cooldown only after basic CSV validation, so users can fix
        # file-format mistakes without waiting five minutes to retry.
        allowed, retry_after = _claim_training_slot(request)
        if not allowed:
            if retry_after is None:
                return None, None, "⚠️ Training is temporarily busy. Please try again later."
            return (
                None,
                None,
                f"⏳ **Please wait about {retry_after} seconds before training again.** "
                "Each client can start one training run every five minutes.",
            )

        # Encode with the frozen encoder in small batches to limit CPU memory use.
        embeddings = []
        batch_count = (len(texts) + 7) // 8
        for batch_index, i in enumerate(range(0, len(texts), 8), start=1):
            progress(
                0.05 + 0.65 * batch_index / batch_count,
                desc=f"Encoding examples ({batch_index}/{batch_count})",
            )
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

        for epoch in range(30):
            progress(
                0.70 + 0.27 * (epoch + 1) / 30,
                desc=f"Training classifier ({epoch + 1}/30)",
            )
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

        # Files in Gradio's temp directory are served as outputs and removed by
        # Blocks.delete_cache after the configured retention period.
        model_path = _create_cache_file(".pt")
        torch.save(custom_head.state_dict(), model_path)

        mapping_path = _create_cache_file(".txt")
        with open(mapping_path, "w", encoding="utf-8") as mapping_file:
            for i, label in enumerate(labels_list):
                mapping_file.write(f"{i}: {label}\n")

        progress(1.0, desc="Preparing your downloads")
        msg = (
            "### ✅ Training complete\n\n"
            f"Trained on **{len(texts)} examples** across **{num_classes} labels** "
            f"for 30 epochs. Final loss: `{final_loss:.4f}`.\n\n"
            "**Label mapping**\n\n"
            + "\n".join(f"- `{i}` → {label}" for i, label in enumerate(labels_list))
            + "\n\nDownload both files below. They remain available temporarily, "
            "so save them before leaving the page."
        )
        return model_path, mapping_path, msg

    except Exception:
        logger.exception("Custom head training failed")
        for path in (locals().get("model_path"), locals().get("mapping_path")):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Could not remove incomplete training artifact: %s", path)
        return (
            None,
            None,
            "❌ **Training couldn't be completed because of an internal error.** "
            "Please try again later. If it continues, contact the site owner.",
        )


# 4. Build the UI using gr.Blocks
with gr.Blocks(
    title="Sama ITSM Router",
    delete_cache=(300, 900),
) as demo:
    gr.Markdown(
        "# Sama ITSM Router\n"
        "Get a suggested support queue for a ticket, or train a small custom router "
        "from your own labeled examples."
    )

    with gr.Tab("Inference Demo"):
        gr.Markdown(
            "Describe the issue in a few sentences. The model suggests one of five queues; "
            "a person should review the result before routing a real ticket."
        )
        ticket_input = gr.Textbox(
            lines=5,
            max_lines=10,
            label="Ticket description",
            placeholder="Example: I cannot access the monthly expense report in the finance portal.",
        )
        with gr.Row():
            route_button = gr.Button("Suggest a queue", variant="primary")
            clear_ticket_button = gr.ClearButton(
                components=[ticket_input], value="Clear ticket"
            )
        gr.Markdown("### Routing suggestion")
        route_output = gr.Markdown("Your routing suggestion will appear here.")
        gr.Examples(
            examples=[
                ["My laptop will not connect to the office Wi-Fi after the latest update."],
                ["I need a copy of the invoice for last month's subscription."],
                ["Can you share details about the company's holiday schedule?"],
            ],
            inputs=ticket_input,
            label="Try an example",
        )
        route_button.click(
            fn=route_ticket,
            inputs=ticket_input,
            outputs=route_output,
            concurrency_id="model-work",
            concurrency_limit=2,
        )

    with gr.Tab("Train Your Own"):
        gr.Markdown(
            "Train a lightweight classifier with your labeled examples. The shared encoder "
            "remains frozen. **Do not upload confidential or personally identifying ticket data.**\n\n"
            "**CSV requirements:** `.csv` file, up to **2 MB** and **300 rows**, with "
            "at least two distinct labels and no blank values."
        )
        with gr.Accordion("CSV format example", open=False):
            gr.Markdown(
                "The column names must be exactly `text` and `label`. Each row is one example."
            )
            gr.Textbox(
                "text,label\n\"Cannot reset my password\",Technical\n"
                "\"Please send my latest invoice\",Financial\n"
                "\"What are the office hours?\",General",
                language="markdown",
                label="Example CSV contents",
            )
        csv_file = gr.File(
            label="Upload your labeled CSV",
            file_types=[".csv"],
            type="filepath",
        )
        with gr.Row():
            train_button = gr.Button("Train my classifier", variant="primary")
            clear_training_button = gr.Button("Clear training form")
        with gr.Row():
            model_download = gr.File(label="1. Model weights (.pt)")
            mapping_download = gr.File(label="2. Label mapping (.txt)")
        train_logs = gr.Markdown(
            "After training, you'll see the result and download links here."
        )
        train_button.click(
            fn=train_custom_head,
            inputs=csv_file,
            outputs=[model_download, mapping_download, train_logs],
            concurrency_id="model-work",
            concurrency_limit=2,
        )
        clear_training_button.click(
            fn=lambda: (None, None, None, "After training, you'll see the result and download links here."),
            outputs=[csv_file, model_download, mapping_download, train_logs],
        )

demo.queue(max_size=10, default_concurrency_limit=2, api_open=False)
demo.launch(server_name="0.0.0.0", server_port=7860, max_file_size="2mb")
