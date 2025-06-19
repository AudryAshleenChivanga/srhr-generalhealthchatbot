import streamlit as st
from transformers import TFGPT2LMHeadModel, AutoTokenizer
import tensorflow as tf
from itertools import groupby
import re

# --- Load tokenizer and model ---
model_path = r"C:/Users/Audry Ashleen/ashlemedbot/distilgpt2_srhr_generalhealth_v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFGPT2LMHeadModel.from_pretrained(model_path)

#  padding token
tokenizer.pad_token = tokenizer.eos_token



# Removing repeated or nearly identical sentences
def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [key for key, _ in groupby(sentences)]
    return " ".join(filtered).strip()

# Detecting and flagging hallucinations or technical gibberish
def is_nonsensical(text):
    hallucination_keywords = [
        "mutation", "binding", "gene", "protein", "tissue", 
        "alpha/beta", "cortisol", "adrenal", "[", "]"
    ]
    return sum(1 for word in hallucination_keywords if word in text.lower()) > 4

# Pink Chat UI Styling 
st.markdown("""
    <style>
    .stApp {
        background-color: #fff0f5;
    }
    h1 {
        color: #d63384;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-align: center;
    }
    .chat-bubble {
        border-radius: 12px;
        padding: 12px 20px;
        margin: 10px 0;
        max-width: 75%;
        font-size: 16px;
        font-family: 'Verdana', sans-serif;
        line-height: 1.4;
        display: inline-block;
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
    </style>
""", unsafe_allow_html=True)

#  Title and Instruction
st.title("AshleMedBot - SRHR & Health Chatbot")
st.markdown("Ask any **health-related** or **SRHR** question below:")

#  Chat History Setup 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from User
user_input = st.text_input("You ", placeholder="e.g. What are signs of PCOS?", key="user_input")

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

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    if "<|sep|>" in decoded_output:
        raw_answer = decoded_output.split("<|sep|>")[-1].strip()
    else:
        raw_answer = decoded_output.strip()

    cleaned_answer = remove_repeated_sentences(raw_answer)

    # Final check for hallucination
    if is_nonsensical(cleaned_answer):
        cleaned_answer = (
            "⚠️ I'm not confident about that answer. "
            "Please consult a medical professional or verified health source."
        )

    st.session_state.chat_history.append(("bot", cleaned_answer))

# Display Chat Bubbles 
for sender, message in st.session_state.chat_history:
    bubble_class = "user-bubble" if sender == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{message}</div>', unsafe_allow_html=True)
