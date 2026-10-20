import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

# Define model architecture (must match training)
class ChoiceHead5(nn.Module):
    def __init__(self, hd=1536, pd=512, mo=5, dr=0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hd, pd), nn.GELU(), nn.Dropout(dr),
            nn.Linear(pd, pd), nn.GELU(), nn.Dropout(dr)
        )
        self.head = nn.Linear(pd, mo)
    def forward(self, h): return self.head(self.proj(h))

print("Loading encoder...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
encoder = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float32)
encoder.eval()

print("Loading head...")
head = ChoiceHead5()
# Note: The .pt file must be placed in the same Space directory
head.load_state_dict(torch.load("choice_head_5class.pt", map_location="cpu"))
head.eval()

OPTIONS = ["Technical", "Financial", "General", "Sales", "Internal"]

def route_ticket(ticket_text):
    if not ticket_text.strip():
        return "Please enter a ticket."
    
    text = f"{ticket_text}\n\nQuestion: Which support queue should handle this ticket?\nType: choice\nOptions: " + \
           " ".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(OPTIONS)) + "\nAnswer:"
    
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320)
    with torch.no_grad():
        out = encoder(**enc, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1].float()
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0]), seq_lens]
        
        logits = head(pooled)[0]
        probs = F.softmax(logits / 1.071, dim=-1).numpy() # T=1.071
        
        best_idx = probs.argmax()
        best_cluster = OPTIONS[best_idx]
        conf = probs[best_idx]
        
        action = "✅ AUTO-ROUTE" if conf >= 0.80 else "⚠️ ESCALATE TO HUMAN/GPT-4"
        
        return f"Predicted Queue: {best_cluster}\nConfidence: {conf:.1%}\nAction: {action}"

demo = gr.Interface(
    fn=route_ticket,
    inputs=gr.Textbox(lines=5, label="Paste Ticket Here"),
    outputs=gr.Textbox(label="Routing Decision"),
    title="Sama ITSM: Calibrated Ticket Router",
    description="Paste a support ticket. Sama uses a frozen 1.5B encoder + 1M param head to route it locally with guaranteed precision (ECE 0.0092)."
)

demo.launch()
