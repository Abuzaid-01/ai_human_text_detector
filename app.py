import streamlit as st
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = RobertaForSequenceClassification.from_pretrained(
        "./models",  # 👈 change here
        local_files_only=True,
        ignore_mismatched_sizes=True 
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "./models",  # 👈 change here
        local_files_only=True
    )
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("AI vs Human Text Classifier 🧠 vs ✍️")
st.write("Paste your text below and find out whether it's written by a Human or AI!")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()

        # Map prediction
        labels = ["Human", "AI"]
        st.success(f"### Prediction: {labels[prediction]} ✨")
        st.write(f"**Confidence**: {probs[0][prediction].item():.2%}")








