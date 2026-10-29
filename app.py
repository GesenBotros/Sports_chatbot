import io
import json

import numpy as np
import streamlit as st
import tensorflow as tf
import torch

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# 1. App configuration
# ==========================================

st.set_page_config(
    page_title="Sports Chatbot",
    page_icon="🏆",
    layout="centered"
)

st.title(" Sports Chatbot")
st.caption("Ask questions about sports or upload a sports image.")


# ==========================================
# 2. System prompt
# ==========================================

SYSTEM_PROMPT = """
You are a sports education assistant.

Rules:
- Answer the user's specific question.
- Football means association football (soccer), not American football.
- Never invent rules, equipment, player counts, scores,
  statistics, or records.
- If you are uncertain, clearly say you are uncertain.
- Keep answers clear and concise.
- If asked who created you, say: "Gesen Botros."
"""


# ==========================================
# 3. Load sports image classifier
# ==========================================

@st.cache_resource
def load_sports_model():
    model = tf.keras.models.load_model(
        "sports_classifier.keras"
    )

    with open(
        "class_names.json",
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model outputs does not match "
            "the number of class names."
        )

    return model, class_names


# ==========================================
# 4. Load chatbot
# ==========================================

@st.cache_resource
def load_chatbot():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model.eval()

    return tokenizer, model


# ==========================================
# 5. Image prediction function
# ==========================================

def predict_sport(image_bytes):
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_resized = image.resize((224, 224))

    image_array = np.array(
        image_resized,
        dtype=np.float32
    )

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = sports_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_sport = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    return predicted_sport, confidence


# ==========================================
# 6. Initialize session state
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 7. Load models
# ==========================================

try:
    sports_model, class_names = load_sports_model()
    tokenizer, model = load_chatbot()

except Exception as error:
    st.error(
        f"Could not load the models: {error}"
    )
    st.stop()


# ==========================================
# 8. Reset conversation
# ==========================================

if st.button("🗑️ Reset Chat"):
    st.session_state.messages = []
    st.rerun()


# ==========================================
# 9. Display conversation history
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("content"):
            st.write(message["content"])

        for image_bytes in message.get("images", []):
            st.image(image_bytes)


# ==========================================
# 10. Chat input with file upload
# ==========================================

prompt = st.chat_input(
    "Ask a sports question or attach an image...",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png"]
)


# ==========================================
# 11. Process the new message
# ==========================================

if prompt:

    user_input = prompt.text or ""
    uploaded_files = prompt.files

    # Store image bytes so they remain available
    # when Streamlit reruns the app.
    uploaded_images = [
        uploaded_file.getvalue()
        for uploaded_file in uploaded_files
    ]

    # Save the user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "images": uploaded_images
    })

    # Display the user message
    with st.chat_message("user"):

        if user_input:
            st.write(user_input)

        for uploaded_file in uploaded_files:
            st.image(
                uploaded_file.getvalue(),
                caption=uploaded_file.name
            )

    # ------------------------------------------
    # 12. Classify uploaded images
    # ------------------------------------------

    for image_bytes in uploaded_images:

        try:
            with st.spinner("Analyzing sports image..."):

                predicted_sport, confidence = predict_sport(
                    image_bytes
                )

            assistant_response = (
                f"🏆 Predicted sport: **{predicted_sport}**\n\n"
                f"Confidence score: **{confidence:.2f}%**"
            )

        except Exception as error:
            assistant_response = (
                f"Could not classify this image: {error}"
            )

        # Save prediction in conversation history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "images": []
        })

        # Display prediction as an assistant message
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    # ------------------------------------------
    # 13. Generate chatbot response to text
    # ------------------------------------------

    if user_input.strip():

        # Build conversation for Qwen.
        # Image bytes are not passed to this text-only model.
        chatbot_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        for message in st.session_state.messages:

            if message["role"] not in ("user", "assistant"):
                continue

            content = message.get("content", "")

            if not content:
                continue

            chatbot_messages.append({
                "role": message["role"],
                "content": content
            })

        inputs = tokenizer.apply_chat_template(
            chatbot_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(model.device)

        # Generate response
        with st.spinner("Thinking..."):

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )

        # Extract newly generated tokens
        input_length = inputs["input_ids"].shape[1]

        new_tokens = outputs[0][input_length:]

        assistant_response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "images": []
        })

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(assistant_response)