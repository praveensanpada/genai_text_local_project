# gen_text_ai_app.py
import streamlit as st
from llama_cpp import Llama
import os

# -----------------------------
# 🧠 Load LLaMA2 GGUF Model
# -----------------------------
MODEL_PATH = "./models/llama2/llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() or 4,
    use_mlock=True,
    verbose=False,
)

# -----------------------------
# 🎨 Streamlit App Styling
# -----------------------------
st.set_page_config(page_title="Gen Text AI", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f4f6f8; }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stButton button {
        font-size: 18px;
        background-color: #005eff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .footer {
        margin-top: 80px;
        font-size: 14px;
        color: #888;
        text-align: center;
    }
    .header {
        background-color: #0f62fe;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧠 Header
# -----------------------------
st.markdown('<div class="header"><h2>🧠 Gen Text AI</h2><p>Your Local AI Text Generator</p></div>', unsafe_allow_html=True)

# -----------------------------
# ✍️ Prompt Input
# -----------------------------
user_prompt = st.text_area("Enter your prompt:", placeholder="e.g. Write a product description for a smart water bottle.", height=150)

if st.button("Generate Text") and user_prompt:
    with st.spinner("Generating..."):
        response = llm(user_prompt, max_tokens=200)
        generated = response["choices"][0]["text"].strip()
        st.markdown("#### 💬 Generated Text")
        st.write(generated)

# -----------------------------
# 🦶 Footer
# -----------------------------
st.markdown("""
    <div class="footer">
        ⓒ 2025 Gen Text AI. Powered by LLaMA2 running locally — no cloud, no API.
    </div>
""", unsafe_allow_html=True)
