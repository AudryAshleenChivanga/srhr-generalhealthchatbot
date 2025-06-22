import streamlit as st
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2LMHeadModel
import tensorflow as tf
from itertools import groupby
import re
from PIL import Image
import os

# --- Hugging Face repo (files should now be in root) ---
HF_REPO = "Audry123/distilgpt2-srhr-generalhealth"

# --- Load tokenizer and TensorFlow model directly from Hugging Face ---
try:
    tokenizer = GPT2Tokenizer.from_pretrained(
        HF_REPO,
        use_fast=False
    )
    model = TFGPT2LMHeadModel.from_pretrained(
        HF_REPO,
        from_pt=False  # <-- set to False since you have a TensorFlow model (tf_model.h5)
    )
    model.trainable = False
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    st.error(f" Failed to load model/tokenizer: {e}")

# --- Utility functions ---
def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [key for key, _ in groupby(sentences)]
    return " ".join(filtered).strip()

def is_nonsensical(text):
    hallucination_keywords = [
        "mutation", "binding", "gene", "protein", "tissue",
        "alpha/beta", "cortisol", "adrenal", "[", "]"
    ]
    return sum(1 for word in hallucination_keywords if word in text.lower()) > 4

# --- Streamlit Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #fff0f5;
        max-width: 700px;
        margin: auto;
        padding: 10px 20px 40px 20px;
        font-family: 'Verdana', sans-serif;
    }
    h1 {
        color: #d63384;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-align: center;
        margin-bottom: 10px;
    }
    .chat-bubble {
        border-radius: 12px;
        padding: 12px 20px;
        margin: 10px 0;
        max-width: 75%;
        font-size: 16px;
        line-height: 1.4;
        display: inline-block;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #ffe3ee;
        color: #4a154b;
        margin-left: auto;
        text-align: right;
    }
    .bot-bubble {
        background-color: #fce4ec;
        color: #4a154b;
        margin-right: auto;
        text-align: left;
    }
    .chat-container {
        margin-top: 20px;
    }
    .caution-text {
        font-size: 14px;
        color: #a83261;
        margin-top: 30px;
        border-top: 1px solid #d63384;
        padding-top: 15px;
        font-style: italic;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Display Chatbot Image ---
img_path = os.path.join("images", "ashlebot.jpg")
if os.path.exists(img_path):
    image = Image.open(img_path)
    st.image(image, caption="AshleBot - Your SRHR Chatbot", use_container_width=True)

# --- Title & Input ---
st.title("AshleMedBot - SRHR & Health Chatbot")
st.markdown("Ask any **health-related** or **SRHR** question below:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You", placeholder="e.g. What are signs of PCOS?", key="user_input")

# --- Chat Logic ---
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    prompt = (
        "You are a helpful and accurate SRHR and health assistant. "
        "If unsure, say 'I'm not sure.'\n\nQ: " + user_input + " <|sep|>"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="tf")

    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3
    )

    decoded_output = tokenizer.decode(output[0].numpy(), skip_special_tokens=True)

    if "<|sep|>" in decoded_output:
        raw_answer = decoded_output.split("<|sep|>")[-1].strip()
    else:
        raw_answer = decoded_output.strip()

    cleaned_answer = remove_repeated_sentences(raw_answer)

    if is_nonsensical(cleaned_answer):
        cleaned_answer = (
            " I'm not confident about that answer. "
            "Please consult a medical professional or verified health source."
        )

    st.session_state.chat_history.append(("bot", cleaned_answer))

# --- Display Chat History ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.chat_history:
    bubble_class = "user-bubble" if sender == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Caution message ---
st.markdown("""
    <div class="caution-text">
         <b>Note:</b> This AI model is still under development and may not provide 100% accurate answers.
        Please do not rely solely on this chatbot for medical decisions. Always consult a qualified healthcare professional for critical health issues.
    </div>
""", unsafe_allow_html=True)
