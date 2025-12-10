import gradio as gr
from faster_whisper import WhisperModel
from llm import llama3_chat_model
from router import route_llm_output

# Load Whisper once
model = WhisperModel("base", device="cpu")

def voice_agent(audio_file):
    # Transcribe speech
    segments, _ = model.transcribe(audio_file)
    transcript = " ".join([seg.text for seg in segments])

    # Get answer from your JSON
    #assistant = llama3_chat_model(transcript)
        # 1. Ask LLM
    llm_response = llama3_chat_model(transcript)

    # 2. Route output to tool if needed
    reply_text = route_llm_output(llm_response)

    # 3. (TTS would go here)
    
    return transcript, llm_response, reply_text

ui = gr.Interface(
    fn=voice_agent,
    inputs=gr.Audio(type="filepath", label="🎤 Speak your question"),
    outputs=[
        gr.Textbox(label="📝 Transcribed User Query"),
        gr.Textbox(label="🧠 Raw LLM Output"),
        gr.Textbox(label="✅ Final Assistant Response")
    ],
    title="Interview Voice Assistant (Whisper + JSON)",
    description="Ask interview questions by speaking. Powered by Whisper + Gradio."
)

ui.launch()
