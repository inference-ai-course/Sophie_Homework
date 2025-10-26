# app.py
import os
import json
from typing import Optional

import gradio as gr
from openai import OpenAI

# Optional: import LangChain (if installed and you want to use it)
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# === Configure OpenAI client to point to Ollama ===
# Either set env vars or hardcode base_url below.
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/v1")

# Create an OpenAI client that will talk to Ollama
client = OpenAI(
    base_url=OLLAMA_BASE,
    api_key="ollama"  # some sdk versions require a key param; Ollama ignores it
)

# Optional: LangChain wrapper configured to talk to Ollama
def make_langchain_model(model_name: str):
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain not installed. pip install langchain to enable.")
    # Many LangChain wrappers pick up OPENAI_API_BASE from env.
    # We'll instruct the user to set it; but we can also construct ChatOpenAI with model_name directly.
    # NOTE: Behavior depends on LangChain version. If this errors, use the Direct mode below.
    return ChatOpenAI(model_name=model_name, openai_api_base=OLLAMA_BASE, openai_api_key="ollama", temperature=0.2)

# === Helper: direct chat call to Ollama via OpenAI SDK ===
def call_ollama_direct(model: str, messages: list, max_tokens: int = 512):
    """
    Calls Ollama using the OpenAI-compatible SDK (openai.OpenAI).
    Returns the text string returned by the model (best attempt).
    """
    # Use the chat.completions endpoint
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    # Debug: if you need to inspect the full object, you can return json.dumps(resp, default=str)
    try:
        # Standard structured response path
        return resp.choices[0].message.content
    except Exception:
        # Fallback for streaming or delta-style responses
        try:
            return resp.choices[0].delta.get("content", "")
        except Exception:
            return str(resp)

# === Example translator helper (very simple) ===
def translate_text(text: str, source_lang: str, target_lang: str, model="llama3"):
    prompt = (
        f"You are a translator. Translate the following text from {source_lang} to {target_lang}.\n\n"
        f"Text:\n{text}\n\n"
        "Return only the translation (no extra commentary)."
    )
    messages = [
        {"role": "system", "content": "You are a helpful translator."},
        {"role": "user", "content": prompt},
    ]
    return call_ollama_direct(model=model, messages=messages, max_tokens=512)

# === Gradio UI callbacks ===
def handle_direct_chat(user_input: str, model_name: str, translate: bool, src_lang: str, tgt_lang: str):
    # Prepare messages for the model (example few-shot + conversation)
    if translate:
        # translate the user input first
        translated = translate_text(user_input, src_lang, tgt_lang, model=model_name)
        user_for_model = f"(Translated input from {src_lang} to {tgt_lang}):\n{translated}"
    else:
        user_for_model = user_input

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and helpful."},
        {"role": "user", "content": user_for_model}
    ]
    answer = call_ollama_direct(model=model_name, messages=messages)
    return answer

def handle_langchain_chat(user_input: str, model_name: str, translate: bool, src_lang: str, tgt_lang: str):
    if not LANGCHAIN_AVAILABLE:
        return "LangChain not installed. Install it with: pip install langchain"
    # Build LangChain model wrapper
    lc_model = make_langchain_model(model_name)
    if translate:
        translated = translate_text(user_input, src_lang, tgt_lang, model=model_name)
        final_input = f"(Translated input from {src_lang} to {tgt_lang}):\n{translated}"
    else:
        final_input = user_input

    # Create simple LangChain chat invocation
    messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=final_input)]
    lc_resp = lc_model(messages)
    # The returned type depends on LangChain version; try common attributes
    try:
        return lc_resp.content
    except Exception:
        try:
            return str(lc_resp)
        except Exception:
            return "LangChain returned an unexpected response; inspect logs."

# === Gradio layout ===
with gr.Blocks(title="Ollama + LangChain → Gradio Proxy Demo") as demo:
    gr.Markdown("# Ollama + LangChain Proxy UI (demo)\n"
                "Use **Direct** mode to call Ollama via the OpenAI SDK, or **LangChain** mode to route through LangChain.\n\n"
                "Make sure Ollama is running (`ollama serve`) and you have pulled a model (e.g. `ollama pull llama3`).")

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(label="User input", placeholder="Enter prompt / question", lines=4)
            model_name = gr.Textbox(label="Model name (Ollama)", value="llama3")
            translate_toggle = gr.Checkbox(label="Translate first (proxy translator)", value=False)
            src_lang = gr.Textbox(label="Source language (when translate)", value="Chinese")
            tgt_lang = gr.Textbox(label="Target language (when translate)", value="English")
            run_direct = gr.Button("Run — Direct Ollama")
            run_langchain = gr.Button("Run — LangChain (optional)")
        with gr.Column(scale=3):
            output = gr.Textbox(label="Model output", lines=12)
            raw_debug = gr.Textbox(label="Raw debug output (optional)", lines=8)

    # Button callbacks
    run_direct.click(fn=lambda inp, m, t, s, g: handle_direct_chat(inp, m, t, s, g),
                     inputs=[user_input, model_name, translate_toggle, src_lang, tgt_lang],
                     outputs=[output])

    run_langchain.click(fn=lambda inp, m, t, s, g: handle_langchain_chat(inp, m, t, s, g),
                        inputs=[user_input, model_name, translate_toggle, src_lang, tgt_lang],
                        outputs=[output])

if __name__ == "__main__":
    # Optional: set environment variable for LangChain (if using it)
    os.environ.setdefault("OPENAI_API_BASE", OLLAMA_BASE)
    os.environ.setdefault("OPENAI_API_KEY", "ollama")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
