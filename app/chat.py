"""Demo em modo chat: SLM_CHECKPOINT=caminho/final.pt streamlit run app/chat.py"""

import os

import streamlit as st
import torch

from slm.chat import render_prompt
from slm.checkpoint import load_model
from slm.generate import stream
from slm.tokenizer import END, ENDOFTEXT

st.set_page_config(page_title="SLM Chat")
st.title("SLM Chat")


@st.cache_resource(show_spinner="Carregando o modelo...")
def get_model(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(path, device)


with st.sidebar:
    checkpoint_path = st.text_input("Checkpoint", os.environ.get("SLM_CHECKPOINT", "checkpoints/sft.pt"))
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.05)
    top_k = st.slider("Top-k", 1, 200, 40)
    max_new_tokens = st.slider("Máximo de tokens", 16, 512, 200, 8)
    if st.button("Limpar conversa"):
        st.session_state.messages = []

if not os.path.exists(checkpoint_path):
    st.error(f"Checkpoint não encontrado: {checkpoint_path}")
    st.stop()

model, tokenizer = get_model(checkpoint_path)
st.caption(f"{model.num_params() / 1e6:.1f}M parâmetros em {next(model.parameters()).device}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escreva uma mensagem"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    def stream_text():
        tokens: list[int] = []
        printed = ""
        for token in stream(
            model,
            render_prompt(st.session_state.messages, tokenizer),
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=frozenset({END, ENDOFTEXT}),
        ):
            tokens.append(token)
            text = tokenizer.decode(tokens)
            # Um caractere pode ocupar mais de um token: espera ele ficar completo.
            if not text.endswith("�"):
                yield text[len(printed) :]
                printed = text

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_text())
    st.session_state.messages.append({"role": "assistant", "content": answer or ""})
