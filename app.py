import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import datetime
import csv

# ─── Page Config ───
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Poppins', sans-serif; }

    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #2d5016 0%, #4a8c2a 50%, #6bbd4e 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p { color: #d4edda; font-size: 1rem; margin: 0.3rem 0 0 0; }

    .result-card {
        background: linear-gradient(135deg, #f0fff0, #e8f5e9);
        border-left: 5px solid #4a8c2a;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .disease-card {
        background: linear-gradient(135deg, #3a1c1c, #4a2020);
        border-left: 5px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #fafafa;
    }
    .disease-card h3 { color: #ff8a80; }
    .healthy-card {
        background: linear-gradient(135deg, #1a3a1a, #1e4a1e);
        border-left: 5px solid #27ae60;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #fafafa;
    }
    .healthy-card h3 { color: #81c784; }
    .info-box {
        background: #1a2332;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #2a3a4a;
        color: #e0e0e0;
    }
    .confidence-high { color: #66bb6a; font-weight: 700; font-size: 1.3rem; }
    .confidence-medium { color: #ffa726; font-weight: 700; font-size: 1.3rem; }
    .confidence-low { color: #ef5350; font-weight: 700; font-size: 1.3rem; }

    .history-row {
        background: #1a2332;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 3px solid #4a8c2a;
        color: #e0e0e0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4a8c2a, #6bbd4e);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a7020, #5aa040);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model and Data ───
@st.cache_resource
def load_model():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)


@st.cache_data
def load_disease_info():
    with open('disease_info.json', 'r', encoding='utf-8') as f:
        return json.load(f)


# ─── Helper Functions ───
def predict_disease(interpreter, image):
    """Run prediction on a single image."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img).astype('float32'), 0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return predictions[0]


def get_confidence_class(confidence):
    """Return CSS class based on confidence level."""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.60:
        return "confidence-medium"
    else:
        return "confidence-low"


def save_to_history(disease, confidence, plant, timestamp):
    """Save prediction to history CSV."""
    os.makedirs('history', exist_ok=True)
    filepath = os.path.join('history', 'predictions.csv')
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Plant', 'Disease', 'Confidence'])
        writer.writerow([timestamp, plant, disease, f"{confidence:.2%}"])


def load_history():
    """Load prediction history from CSV."""
    filepath = os.path.join('history', 'predictions.csv')
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_chatbot_response(disease_name, question, lang_code, disease_data):
    """Generate chatbot response using disease info + OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        lang_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
        lang = lang_names.get(lang_code, "English")

        system_prompt = f"""You are an expert plant pathologist assistant. The user has a plant diagnosed with: {disease_name}.

Here is the known information about this disease:
- Cause: {disease_data.get('cause', 'N/A')}
- Symptoms: {disease_data.get('symptoms', 'N/A')}
- Treatment: {disease_data.get('treatment', 'N/A')}
- Prevention: {disease_data.get('prevention', 'N/A')}

Respond in {lang}. Be helpful, concise, and practical. If asked something outside your knowledge, say so honestly."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    except ImportError:
        return "⚠️ Please install openai package: `pip install openai`"
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            return "⚠️ Please set your OPENAI_API_KEY environment variable. Run: `export OPENAI_API_KEY=your_key_here`"
        return f"⚠️ Chatbot error: {error_msg}"

# ─── Language Labels ───
LABELS = {
    "en": {
        "title": "🌿 Plant Disease Detector",
        "subtitle": "AI-powered plant disease detection with treatment suggestions",
        "upload": "Upload a leaf image",
        "analyze": "🔍 Analyze Leaf",
        "result": "Diagnosis Result",
        "confidence": "Confidence",
        "plant": "Plant",
        "disease": "Disease",
        "cause": "🔬 Cause",
        "symptoms": "🩺 Symptoms",
        "treatment": "💊 Treatment",
        "prevention": "🛡️ Prevention",
        "healthy_msg": "Your plant looks healthy! Keep up the good care.",
        "history": "📋 Prediction History",
        "no_history": "No predictions yet. Upload an image to get started!",
        "chatbot": "🤖 Ask the Plant Doctor",
        "chat_placeholder": "Ask a question about this disease...",
        "chat_send": "Send",
        "language": "🌐 Language",
        "clear_history": "🗑️ Clear History",
        "sidebar_info": "About",
        "sidebar_desc": "This app uses deep learning (EfficientNetB0) to detect plant diseases from leaf images. It supports Pepper, Potato, and Tomato diseases.",
        "accuracy": "Model Accuracy: 96.3%",
        "low_confidence": "⚠️ Low confidence prediction. Try uploading a clearer image."
    },
    "hi": {
        "title": "🌿 पौधों की बीमारी पहचानक",
        "subtitle": "AI-संचालित पौधों की बीमारी पहचान और उपचार सुझाव",
        "upload": "एक पत्ती की तस्वीर अपलोड करें",
        "analyze": "🔍 पत्ती का विश्लेषण करें",
        "result": "निदान परिणाम",
        "confidence": "विश्वास स्तर",
        "plant": "पौधा",
        "disease": "बीमारी",
        "cause": "🔬 कारण",
        "symptoms": "🩺 लक्षण",
        "treatment": "💊 उपचार",
        "prevention": "🛡️ रोकथाम",
        "healthy_msg": "आपका पौधा स्वस्थ दिखता है! अच्छी देखभाल जारी रखें।",
        "history": "📋 पूर्वानुमान इतिहास",
        "no_history": "अभी तक कोई पूर्वानुमान नहीं। शुरू करने के लिए एक तस्वीर अपलोड करें!",
        "chatbot": "🤖 पौधे के डॉक्टर से पूछें",
        "chat_placeholder": "इस बीमारी के बारे में कोई सवाल पूछें...",
        "chat_send": "भेजें",
        "language": "🌐 भाषा",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "sidebar_info": "जानकारी",
        "sidebar_desc": "यह ऐप पत्ती की तस्वीरों से पौधों की बीमारियों का पता लगाने के लिए डीप लर्निंग (EfficientNetB0) का उपयोग करता है। यह शिमला मिर्च, आलू और टमाटर की बीमारियों को सपोर्ट करता है।",
        "accuracy": "मॉडल सटीकता: 96.3%",
        "low_confidence": "⚠️ कम विश्वास स्तर। कृपया एक स्पष्ट तस्वीर अपलोड करें।"
    },
    "kn": {
        "title": "🌿 ಸಸ್ಯ ರೋಗ ಪತ್ತೆಕಾರ",
        "subtitle": "AI-ಚಾಲಿತ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಸಲಹೆಗಳು",
        "upload": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "analyze": "🔍 ಎಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "result": "ರೋಗನಿರ್ಣಯ ಫಲಿತಾಂಶ",
        "confidence": "ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "plant": "ಸಸ್ಯ",
        "disease": "ರೋಗ",
        "cause": "🔬 ಕಾರಣ",
        "symptoms": "🩺 ಲಕ್ಷಣಗಳು",
        "treatment": "💊 ಚಿಕಿತ್ಸೆ",
        "prevention": "🛡️ ತಡೆಗಟ್ಟುವಿಕೆ",
        "healthy_msg": "ನಿಮ್ಮ ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ! ಉತ್ತಮ ಆರೈಕೆ ಮುಂದುವರಿಸಿ.",
        "history": "📋 ಪೂರ್ವಾನುಮಾನ ಇತಿಹಾಸ",
        "no_history": "ಇನ್ನೂ ಯಾವುದೇ ಪೂರ್ವಾನುಮಾನಗಳಿಲ್ಲ. ಪ್ರಾರಂಭಿಸಲು ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ!",
        "chatbot": "🤖 ಸಸ್ಯ ವೈದ್ಯರನ್ನು ಕೇಳಿ",
        "chat_placeholder": "ಈ ರೋಗದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ...",
        "chat_send": "ಕಳುಹಿಸಿ",
        "language": "🌐 ಭಾಷೆ",
        "clear_history": "🗑️ ಇತಿಹಾಸ ತೆರವುಗೊಳಿಸಿ",
        "sidebar_info": "ಮಾಹಿತಿ",
        "sidebar_desc": "ಈ ಅಪ್ಲಿಕೇಶನ್ ಎಲೆಯ ಚಿತ್ರಗಳಿಂದ ಸಸ್ಯ ರೋಗಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಡೀಪ್ ಲರ್ನಿಂಗ್ (EfficientNetB0) ಬಳಸುತ್ತದೆ. ಇದು ದೊಡ್ಡ ಮೆಣಸಿನಕಾಯಿ, ಆಲೂಗಡ್ಡೆ ಮತ್ತು ಟೊಮೆಟೊ ರೋಗಗಳನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ.",
        "accuracy": "ಮಾದರಿ ನಿಖರತೆ: 96.3%",
        "low_confidence": "⚠️ ಕಡಿಮೆ ವಿಶ್ವಾಸ ಮಟ್ಟ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
    }
}


# ─── Initialize ───
interpreter = load_model()
class_names = load_class_names()
disease_info = load_disease_info()

# ─── Sidebar ───
with st.sidebar:
    lang = st.selectbox(
        LABELS["en"]["language"],
        options=["en", "hi", "kn"],
        format_func=lambda x: {"en": "English", "hi": "हिंदी (Hindi)", "kn": "ಕನ್ನಡ (Kannada)"}[x],
        index=0
    )
    L = LABELS[lang]

    st.markdown("---")
    st.markdown(f"### {L['sidebar_info']}")
    st.markdown(L['sidebar_desc'])
    st.markdown(f"**{L['accuracy']}**")

    st.markdown("---")
    st.markdown("### 🌱 Supported Plants")
    st.markdown("- 🫑 Bell Pepper (2 classes)")
    st.markdown("- 🥔 Potato (3 classes)")
    st.markdown("- 🍅 Tomato (10 classes)")

# ─── Header ───
st.markdown(f"""
<div class="main-header">
    <h1>{L['title']}</h1>
    <p>{L['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ───
col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader(L['upload'], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(L['analyze'], use_container_width=True):
            with st.spinner("Analyzing..."):
                predictions = predict_disease(interpreter, image)
                predicted_idx = np.argmax(predictions)
                confidence = predictions[predicted_idx]
                predicted_class = class_names[predicted_idx]

                # Store in session state
                st.session_state['prediction'] = {
                    'class': predicted_class,
                    'confidence': confidence,
                    'predictions': predictions
                }

                # Save to history
                info = disease_info.get(predicted_class, {}).get(lang, {})
                save_to_history(
                    info.get('disease', predicted_class),
                    confidence,
                    info.get('plant', 'Unknown'),
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

with col2:
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        predicted_class = pred['class']
        confidence = pred['confidence']

        info = disease_info.get(predicted_class, {}).get(lang, {})
        is_healthy = 'healthy' in predicted_class.lower()

        # ─── Result Card ───
        card_class = "healthy-card" if is_healthy else "disease-card"
        conf_class = get_confidence_class(confidence)

        st.markdown(f"### {L['result']}")

        st.markdown(f"""
        <div class="{card_class}">
            <h3>🌱 {info.get('plant', 'Unknown')} — {info.get('disease', predicted_class)}</h3>
            <p>{L['confidence']}: <span class="{conf_class}">{confidence:.1%}</span></p>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 0.60:
            st.warning(L['low_confidence'])

        if is_healthy:
            st.success(L['healthy_msg'])
        else:
            # ─── Treatment Info ───
            st.markdown(f"#### {L['cause']}")
            st.markdown(f"""<div class="info-box">{info.get('cause', 'N/A')}</div>""", unsafe_allow_html=True)

            st.markdown(f"#### {L['symptoms']}")
            st.markdown(f"""<div class="info-box">{info.get('symptoms', 'N/A')}</div>""", unsafe_allow_html=True)

            st.markdown(f"#### {L['treatment']}")
            st.markdown(f"""<div class="info-box">{info.get('treatment', 'N/A')}</div>""", unsafe_allow_html=True)

            st.markdown(f"#### {L['prevention']}")
            st.markdown(f"""<div class="info-box">{info.get('prevention', 'N/A')}</div>""", unsafe_allow_html=True)

        # ─── Top 3 Predictions ───
        st.markdown("---")
        st.markdown("#### Top 3 Predictions")
        top3_idx = np.argsort(pred['predictions'])[::-1][:3]
        for idx in top3_idx:
            name = class_names[idx]
            prob = pred['predictions'][idx]
            display_name = disease_info.get(name, {}).get(lang, {}).get('disease', name)
            plant_name = disease_info.get(name, {}).get(lang, {}).get('plant', '')
            st.progress(float(prob), text=f"{plant_name} — {display_name}: {prob:.1%}")

# ─── Chatbot Section ───
st.markdown("---")
st.markdown(f"### {L['chatbot']}")

if 'prediction' in st.session_state:
    pred = st.session_state['prediction']
    predicted_class = pred['class']
    info = disease_info.get(predicted_class, {}).get(lang, {})

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_question = st.chat_input(L['chat_placeholder'])

    if user_question:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chatbot_response(
                    info.get('disease', predicted_class),
                    user_question,
                    lang,
                    info
                )
                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
else:
    st.info("Upload and analyze an image first to ask questions about the disease.")

# ─── History Section ───
st.markdown("---")
st.markdown(f"### {L['history']}")

history = load_history()

if history:
    col_h1, col_h2 = st.columns([4, 1])
    with col_h2:
        if st.button(L['clear_history']):
            filepath = os.path.join('history', 'predictions.csv')
            if os.path.exists(filepath):
                os.remove(filepath)
                st.rerun()

    # Display history in reverse (newest first)
    for entry in reversed(history[-10:]):  # Show last 10
        st.markdown(f"""
        <div class="history-row">
            <strong>📅 {entry.get('Timestamp', 'N/A')}</strong> &nbsp;|&nbsp;
            🌱 {entry.get('Plant', 'N/A')} &nbsp;|&nbsp;
            🦠 {entry.get('Disease', 'N/A')} &nbsp;|&nbsp;
            📊 {entry.get('Confidence', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info(L['no_history'])

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🌿 Plant Disease Detection System | Final Year Project | Powered by EfficientNetB0 + TFLite</p>
</div>
""", unsafe_allow_html=True)
