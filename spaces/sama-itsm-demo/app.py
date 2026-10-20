import ipaddress
import json
import logging
import os
import tempfile
import threading
import time
import zipfile

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
from sama import __version__ as SAMA_VERSION
from sama.encoder import FrozenEncoder
from sama.format import format_choice
from sama.heads import ChoiceHead

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
ENCODER_ID = "Qwen/Qwen2.5-1.5B"
ENCODER_DTYPE = "float32"
encoder = FrozenEncoder(ENCODER_ID, device="cpu", dtype=torch.float32)


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


CUSTOM_HEAD_INSTRUCTIONS = "Which support queue should handle this ticket?"
MAX_CUSTOM_CLASSES = 8


def _build_custom_head():
    # Use Sama's native choice-head layout so DecisionEngine can load the bundle.
    return ChoiceHead(hidden_dim=encoder.hidden_dim, max_options=MAX_CUSTOM_CLASSES)


def _encode_tickets(
    ticket_texts,
    labels,
    batch_size=8,
    progress_fn=None,
    progress_start=0.0,
    progress_span=0.0,
    progress_desc="Encoding examples",
):
    formatted = [
        format_choice(
            text,
            {"instructions": CUSTOM_HEAD_INSTRUCTIONS, "options": labels},
        )
        for text in ticket_texts
    ]
    batches = []
    starts = range(0, len(formatted), batch_size)
    for batch_index, start in enumerate(starts, start=1):
        batches.append(
            encoder.encode_last_token(formatted[start : start + batch_size], max_length=320)
        )
        if progress_fn:
            progress_fn(
                progress_start + progress_span * batch_index / len(starts),
                desc=f"{progress_desc} ({batch_index}/{len(starts)})",
            )
    return torch.cat(batches, dim=0).float()


# 2. Inference Function
def route_ticket(ticket_text):
    if not ticket_text or not ticket_text.strip():
        return "⚠️ **Enter a ticket description to get a routing suggestion.**"

    with torch.no_grad():
        pooled = _encode_tickets([ticket_text], OPTIONS)
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


def _load_custom_head(trained_model):
    if not trained_model:
        raise ValueError("Train a classifier first.")
    model_path = trained_model.get("model_path")
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError("This trained model has expired. Train it again.")
    labels = trained_model.get("labels", [])
    if not 2 <= len(labels) <= MAX_CUSTOM_CLASSES:
        raise ValueError("The saved classifier has an invalid label mapping.")
    custom_head = _build_custom_head()
    custom_head.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    custom_head.eval()
    return custom_head, labels


def test_custom_classifier(ticket_text, trained_model):
    if not ticket_text or not ticket_text.strip():
        return "⚠️ **Enter a ticket description to test your classifier.**"
    try:
        custom_head, labels = _load_custom_head(trained_model)
        with torch.no_grad():
            pooled = _encode_tickets([ticket_text.strip()], labels)
            probabilities = F.softmax(custom_head(pooled)[0, : len(labels)], dim=-1)
            best_id = int(probabilities.argmax().item())
            confidence = float(probabilities[best_id].item())
        return (
            f"### Custom prediction: {labels[best_id]}\n\n"
            f"**Model score:** {confidence:.1%}\n\n"
            "This result comes from your uploaded training examples. Review it before use."
        )
    except (ValueError, FileNotFoundError) as exc:
        return f"ℹ️ **{exc}**"
    except Exception:
        logger.exception("Testing a custom classifier failed")
        return "❌ **Could not test this classifier.** Try training it again."


def evaluate_custom_classifier(file, trained_model, progress=gr.Progress(track_tqdm=True)):
    if file is None:
        return "⚠️ **Upload a labeled test CSV first.**"
    if not trained_model:
        return "ℹ️ **Train a classifier in the Train Your Own tab first.**"

    try:
        file_path = file.name if hasattr(file, "name") else os.fspath(file)
        if os.path.splitext(file_path)[1].lower() != ".csv":
            return "⚠️ **The test file must have a `.csv` extension.**"
        if os.path.getsize(file_path) > 2 * 1024 * 1024:
            return "⚠️ **The test CSV is larger than 2 MB.** Reduce its size and try again."
        try:
            df = pd.read_csv(file_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            return "⚠️ **We couldn't read this as a valid CSV.** Check the file and try again."
        if "text" not in df.columns or "label" not in df.columns:
            return "⚠️ **The test CSV needs columns named `text` and `label`.**"
        if not 1 <= len(df) <= 300:
            return "⚠️ **The test CSV must contain between 1 and 300 data rows.**"
        if df[["text", "label"]].isna().any().any():
            return "⚠️ **Some `text` or `label` cells are empty.** Fill them in and try again."

        texts = df["text"].astype(str).str.strip().tolist()
        targets = df["label"].astype(str).str.strip().tolist()
        if any(not text for text in texts) or any(not label for label in targets):
            return "⚠️ **Every test row needs non-empty text and label values.**"

        custom_head, labels = _load_custom_head(trained_model)
        label2id = {label: index for index, label in enumerate(labels)}
        unknown_labels = sorted(set(targets) - set(label2id))
        if unknown_labels:
            shown = ", ".join(f"`{label}`" for label in unknown_labels[:5])
            return (
                "⚠️ **The test CSV contains labels that were not in the training data:** "
                f"{shown}. Use the exact same label names."
            )

        with torch.no_grad():
            features = _encode_tickets(
                texts,
                labels,
                batch_size=8,
                progress_fn=progress,
                progress_start=0.05,
                progress_span=0.75,
                progress_desc="Encoding test examples",
            )
            logits = custom_head(features)[:, : len(labels)]
            predictions = logits.argmax(dim=-1).tolist()
        target_ids = [label2id[label] for label in targets]
        accuracy = sum(p == y for p, y in zip(predictions, target_ids)) / len(target_ids)

        class_f1 = []
        evaluated_class_ids = sorted(set(target_ids) | set(predictions))
        for class_id in evaluated_class_ids:
            true_positive = sum(p == class_id and y == class_id for p, y in zip(predictions, target_ids))
            false_positive = sum(p == class_id and y != class_id for p, y in zip(predictions, target_ids))
            false_negative = sum(p != class_id and y == class_id for p, y in zip(predictions, target_ids))
            denominator = 2 * true_positive + false_positive + false_negative
            class_f1.append(2 * true_positive / denominator if denominator else 0.0)
        macro_f1 = sum(class_f1) / len(class_f1)

        return (
            "### Evaluation complete\n\n"
            f"- **Examples evaluated:** {len(target_ids)}\n"
            f"- **Accuracy:** {accuracy:.1%}\n"
            f"- **Macro-F1:** {macro_f1:.3f}\n\n"
            "Use a test CSV that was not used to train the classifier for a more useful estimate."
        )
    except (ValueError, OSError) as exc:
        return f"⚠️ **Could not evaluate this test file:** {exc}"
    except Exception:
        logger.exception("Custom classifier evaluation failed")
        return "❌ **Evaluation couldn't be completed because of an internal error.** Try again later."


# 3. Training Function
def train_custom_head(
    file,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    if file is None:
        return None, "⚠️ **Choose a CSV file first.**", None

    try:
        file_path = file.name if hasattr(file, "name") else os.fspath(file)
    except (TypeError, ValueError):
        return None, "⚠️ **The uploaded file could not be read. Please upload it again.**", None

    if os.path.splitext(file_path)[1].lower() != ".csv":
        return None, "⚠️ **This file does not have a `.csv` extension.**", None

    try:
        if os.path.getsize(file_path) > 2 * 1024 * 1024:
            return None, "⚠️ **The CSV is larger than 2 MB.** Reduce its size and try again.", None
    except OSError:
        return None, "⚠️ **The uploaded file is no longer available. Please upload it again.**", None

    try:
        progress(0.02, desc="Reading and checking your CSV")
        try:
            df = pd.read_csv(file_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            return None, "⚠️ **We couldn't read this as a valid CSV.** Check the file and try again.", None
        if "text" not in df.columns or "label" not in df.columns:
            return None, (
                "⚠️ **Your CSV needs columns named `text` and `label`.** "
                "Column names are case-sensitive."
            ), None
        if len(df) == 0:
            return None, "⚠️ **Your CSV has a header but no data rows.** Add at least two examples.", None
        if len(df) > 300:
            return (
                None,
                f"⚠️ **Your CSV has {len(df)} rows; the limit is 300.** "
                "Use the training script for a larger dataset.",
                None,
            )
        if df[["text", "label"]].isna().any().any():
            return None, "⚠️ **Some `text` or `label` cells are empty.** Fill them in and upload again.", None

        texts = df["text"].astype(str).str.strip().tolist()
        labels_list = df["label"].astype(str).str.strip().tolist()
        if any(not text for text in texts):
            return None, "⚠️ **Every row needs ticket text.** Remove blank `text` cells and try again.", None
        if any(not label for label in labels_list):
            return None, "⚠️ **Every row needs a label.** Remove blank `label` cells and try again.", None

        # Keep the class order stable in the order labels first appear in the CSV.
        labels_list = list(dict.fromkeys(labels_list))
        if len(labels_list) < 2:
            return None, "⚠️ **At least two different labels are required** to train a classifier.", None
        if len(labels_list) > MAX_CUSTOM_CLASSES:
            return None, (
                f"⚠️ **Your CSV has {len(labels_list)} labels; Sama choice classifiers support "
                f"up to {MAX_CUSTOM_CLASSES}.** Combine labels and try again."
            ), None
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
                return None, "⚠️ Training is temporarily busy. Please try again later.", None
            return (
                None,
                f"⏳ **Please wait about {retry_after} seconds before training again.** "
                "Each client can start one training run every five minutes.",
                None,
            )

        # Encode with the frozen encoder in small batches to limit CPU memory use.
        with torch.no_grad():
            X = _encode_tickets(
                texts,
                labels_list,
                batch_size=8,
                progress_fn=progress,
                progress_start=0.05,
                progress_span=0.65,
            )

        # Train Sama's native choice head, masking unused choice slots.
        custom_head = _build_custom_head()
        opt = torch.optim.AdamW(custom_head.parameters(), lr=2e-5)
        final_loss = 0.0

        for epoch in range(30):
            progress(
                0.70 + 0.27 * (epoch + 1) / 30,
                desc=f"Training classifier ({epoch + 1}/30)",
            )
            custom_head.train()
            opt.zero_grad()
            logits = custom_head(X)[:, :num_classes]
            ce = F.cross_entropy(logits, labels)
            p = F.softmax(logits, dim=-1)
            oh = F.one_hot(labels, num_classes).float()
            brier = ((p - oh) ** 2).sum(-1).mean()
            loss = ce + 0.5 * brier
            loss.backward()
            opt.step()
            final_loss = loss.item()

        custom_head.eval()

        # Keep the working state and ZIP in Gradio's managed cache. The working
        # head is session-referenced; the ZIP is the user's portable download.
        model_path = _create_cache_file(".pt")
        torch.save(custom_head.state_dict(), model_path)
        bundle_path = _create_cache_file(".zip")
        config = {
            "format_version": "1.0.0",
            "package_version": SAMA_VERSION,
            "encoder": ENCODER_ID,
            "encoder_dtype": ENCODER_DTYPE,
            "hidden_dim": encoder.hidden_dim,
            "labels": labels_list,
            "head_architecture": {
                "type": "choice",
                "hidden_dim": encoder.hidden_dim,
                "projection_dim": 512,
                "max_options": MAX_CUSTOM_CLASSES,
                "dropout": 0.15,
            },
            "heads": {
                "choice": {
                    "file": "custom_head.pt",
                    "format_version": "1.0.0",
                    "calibration": {"temperature": 1.0},
                }
            },
            "training": {
                "examples": len(texts),
                "classes": num_classes,
                "epochs": 30,
                "loss": "cross_entropy + 0.5 * brier_score",
                "instructions": CUSTOM_HEAD_INSTRUCTIONS,
            },
        }
        bundle_readme = '''# Custom Sama ITSM classifier

This bundle contains a custom choice head trained on your labeled examples.
The frozen encoder is **not** included; Sama downloads `Qwen/Qwen2.5-1.5B`
the first time it is used. Keep all bundle files together.

## Install Sama

Install the Sama version that supports the `encoder_dtype` field in this config:

```bash
python -m pip install "git+https://github.com/vivek5200/sama.git"
```

## Load and predict

```python
import json
from sama import DecisionEngine

bundle_dir = "./custom_classifier_bundle"  # directory containing this README
with open(f"{bundle_dir}/config.json", encoding="utf-8") as f:
    config = json.load(f)

engine = DecisionEngine.from_pretrained(bundle_dir, device="cpu")
ticket = "My expense reimbursement has not arrived."
result = engine.decide(
    state=ticket,
    questions={
        "routing": {
            "type": "choice",
            "instructions": config["training"]["instructions"],
            "options": config["labels"],
        }
    },
)
decision = result.decisions["routing"]
print("Predicted label:", decision.answer)
print("Model score:", decision.confidence)
```

`custom_head.pt` contains the head weights, `label_mapping.txt` lists the
integer-to-label mapping, and `config.json` records the encoder, head, and
training settings. The model score is not a guarantee of correctness; validate
the classifier on examples that were not used for training.
'''
        with tempfile.TemporaryDirectory(prefix="sama-bundle-") as bundle_dir:
            with open(os.path.join(bundle_dir, "config.json"), "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2, ensure_ascii=False)
            with open(os.path.join(bundle_dir, "label_mapping.txt"), "w", encoding="utf-8") as mapping_file:
                for class_id, label in enumerate(labels_list):
                    mapping_file.write(f"{class_id}: {label}\n")
            with open(os.path.join(bundle_dir, "README.md"), "w", encoding="utf-8") as readme_file:
                readme_file.write(bundle_readme)
            with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(model_path, "custom_head.pt")
                for filename in ("config.json", "label_mapping.txt", "README.md"):
                    bundle.write(os.path.join(bundle_dir, filename), filename)

        progress(1.0, desc="Preparing your downloads")
        msg = (
            "### ✅ Training complete\n\n"
            f"Trained on **{len(texts)} examples** across **{num_classes} labels** "
            f"for 30 epochs. Final loss: `{final_loss:.4f}`.\n\n"
            "**Label mapping**\n\n"
            + "\n".join(f"- `{i}` → {label}" for i, label in enumerate(labels_list))
            + "\n\nYou can test it in the **Evaluate** tab. Download the ZIP to keep the model, "
            "label mapping, configuration, and usage instructions together."
        )
        return bundle_path, msg, {"model_path": model_path, "labels": labels_list}

    except Exception:
        logger.exception("Custom head training failed")
        for path in (
            locals().get("model_path"),
            locals().get("bundle_path"),
        ):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Could not remove incomplete training artifact: %s", path)
        return (
            None,
            "❌ **Training couldn't be completed because of an internal error.** "
            "Please try again later. If it continues, contact the site owner.",
            None,
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
    gr.Markdown(
        "[GitHub repository](https://github.com/vivek5200/sama) · "
        "[Qwen2.5-1.5B on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B)"
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
                value=(
                    "text,label\n\"Cannot reset my password\",Technical\n"
                    "\"Please send my latest invoice\",Financial\n"
                    "\"What are the office hours?\",General"
                ),
                lines=4,
                label="Example CSV contents",
                interactive=False,
            )
        csv_file = gr.File(
            label="Upload your labeled CSV",
            file_types=[".csv"],
            type="filepath",
        )
        with gr.Row():
            train_button = gr.Button("Train my classifier", variant="primary")
            clear_training_button = gr.Button("Clear training form")
        bundle_download = gr.File(label="Download custom classifier bundle (.zip)")
        train_logs = gr.Markdown(
            "After training, you'll see the result and download links here."
        )
        trained_head_state = gr.State(None)
        train_button.click(
            fn=train_custom_head,
            inputs=csv_file,
            outputs=[bundle_download, train_logs, trained_head_state],
            concurrency_id="model-work",
            concurrency_limit=2,
        )

        gr.Markdown("### Test your trained classifier")
        gr.Markdown(
            "Enter a new ticket to see which label your custom model predicts. "
            "Train a model above first; testing uses the same shared encoder."
        )
        custom_test_input = gr.Textbox(
            lines=3,
            label="Ticket to classify",
            placeholder="Example: My expense reimbursement has not arrived.",
        )
        test_custom_button = gr.Button("Test custom classifier", variant="primary")
        custom_test_output = gr.Markdown(
            "Your custom model prediction will appear here."
        )
        test_custom_button.click(
            fn=test_custom_classifier,
            inputs=[custom_test_input, trained_head_state],
            outputs=custom_test_output,
            concurrency_id="model-work",
            concurrency_limit=2,
        )

        clear_training_button.click(
            fn=lambda: (
                None,
                None,
                "After training, you'll see the result and download links here.",
                None,
                "Your custom model prediction will appear here.",
                None,
            ),
            outputs=[
                csv_file,
                bundle_download,
                train_logs,
                custom_test_input,
                custom_test_output,
                trained_head_state,
            ],
        )

    with gr.Tab("Evaluate"):
        gr.Markdown(
            "Upload labeled examples that were **not used for training** to measure how well "
            "your latest custom classifier generalizes. The CSV must have `text` and `label` "
            "columns, use the exact labels from training, and contain up to 300 rows / 2 MB."
        )
        evaluation_csv = gr.File(
            label="Upload held-out test CSV",
            file_types=[".csv"],
            type="filepath",
        )
        with gr.Row():
            evaluate_button = gr.Button("Evaluate classifier", variant="primary")
            clear_evaluation_button = gr.ClearButton(
                components=[evaluation_csv], value="Clear test CSV"
            )
        evaluation_output = gr.Markdown(
            "Train a classifier first, then upload a labeled test CSV to see its metrics."
        )
        evaluate_button.click(
            fn=evaluate_custom_classifier,
            inputs=[evaluation_csv, trained_head_state],
            outputs=evaluation_output,
            concurrency_id="model-work",
            concurrency_limit=2,
        )

demo.queue(max_size=10, default_concurrency_limit=2, api_open=False)
demo.launch(server_name="0.0.0.0", server_port=7860, max_file_size="2mb")
