"""Sama FastAPI server — serve calibrated decisions over HTTP."""
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("FastAPI not installed. Run: pip install sama[server]")


class DecideRequest(BaseModel):
    state: str
    questions: Dict[str, Any]


# Global engine instance (loaded at startup)
_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup."""
    global _engine
    from sama import DecisionEngine

    model = os.environ.get("SAMA_MODEL", "Vivek1225/sama-qwen-1.5b-v0.2")
    _engine = DecisionEngine.from_pretrained(model)
    yield
    _engine = None


app = FastAPI(
    title="Sama Decision Engine",
    description="Calibrated decision engine API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decide")
def decide(req: DecideRequest):
    result = _engine.decide(state=req.state, questions=req.questions)
    return result.model_dump()


_HTML_FORM = """<!DOCTYPE html>
<html>
<head><title>Sama</title>
<style>
  body { font-family: system-ui; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  textarea, input, button { width: 100%%; padding: .5rem; margin: .25rem 0; box-sizing: border-box; }
  button { background: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer; }
  pre { background: #f1f5f9; padding: 1rem; border-radius: 4px; overflow-x: auto; }
</style></head>
<body>
<h1>Sama Decision Engine</h1>
<form id="f">
  <label>State (context)</label>
  <textarea id="state" rows="3">What is the chemical symbol for gold?</textarea>
  <label>Instructions</label>
  <input id="instr" value="Which of the following is the correct answer?">
  <label>Options (comma-separated)</label>
  <input id="opts" value="Au, Ag, Gd, Go">
  <button type="submit">Decide</button>
</form>
<pre id="out"></pre>
<script>
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const opts = document.getElementById('opts').value.split(',').map(s=>s.trim());
  const body = {
    state: document.getElementById('state').value,
    questions: { q: { type: 'choice',
      instructions: document.getElementById('instr').value, options: opts }}
  };
  const r = await fetch('/decide', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
};
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML_FORM
