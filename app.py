import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
import asyncio
import tempfile
import os
import json

# ============================================
# LOAD ENV
# ============================================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================
# GEMINI CONFIG
# ============================================
genai.configure(api_key=API_KEY)

MODEL = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ============================================
# TESSERACT PATH
# ============================================
import platform
import pytesseract

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

# ============================================
# STREAMLIT CONFIG
# ============================================
st.set_page_config(
    page_title="AudioVision AI",
    layout="wide"
)

# ============================================
# LANGUAGES
# ============================================
OCR_LANGUAGES = {
    "English": "eng",
    "Hindi": "hin",
    "Marathi": "mar",
    "Gujarati": "guj",
    "Punjabi": "pan",
    "Bengali": "ben",
    "Tamil": "tam",
    "Telugu": "tel",
    "Kannada": "kan",
    "Malayalam": "mal",
    "Urdu": "urd",
    "Sanskrit": "san",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Russian": "rus",
    "Chinese Simplified": "chi_sim",
    "Chinese Traditional": "chi_tra",
    "Japanese": "jpn",
    "Korean": "kor",
    "Arabic": "ara",
    "Turkish": "tur",
    "Dutch": "nld",
    "Greek": "ell",
    "Polish": "pol",
    "Thai": "tha",
    "Vietnamese": "vie",
    "Indonesian": "ind",
    "Malay": "msa",
    "Czech": "ces",
    "Danish": "dan",
    "Finnish": "fin",
    "Hungarian": "hun",
    "Norwegian": "nor",
    "Romanian": "ron",
    "Slovak": "slk",
    "Swedish": "swe",
    "Ukrainian": "ukr"
}


TTS_VOICES = {
    "English": "en-US-AriaNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "Marathi": "mr-IN-AarohiNeural",
    "Gujarati": "gu-IN-DhwaniNeural",
    "Punjabi": "pa-IN-GurleenNeural",
    "Bengali": "bn-IN-TanishaaNeural",
    "Tamil": "ta-IN-PallaviNeural",
    "Telugu": "te-IN-ShrutiNeural",
    "Kannada": "kn-IN-SapnaNeural",
    "Malayalam": "ml-IN-SobhanaNeural",
    "Urdu": "ur-PK-UzmaNeural",
    "Spanish": "es-ES-ElviraNeural",
    "French": "fr-FR-DeniseNeural",
    "German": "de-DE-KatjaNeural",
    "Italian": "it-IT-ElsaNeural",
    "Portuguese": "pt-BR-FranciscaNeural",
    "Russian": "ru-RU-SvetlanaNeural",
    "Chinese Simplified": "zh-CN-XiaoxiaoNeural",
    "Chinese Traditional": "zh-TW-HsiaoChenNeural",
    "Japanese": "ja-JP-NanamiNeural",
    "Korean": "ko-KR-SunHiNeural",
    "Arabic": "ar-SA-ZariyahNeural",
    "Turkish": "tr-TR-EmelNeural",
    "Dutch": "nl-NL-ColetteNeural",
    "Greek": "el-GR-AthinaNeural",
    "Polish": "pl-PL-ZofiaNeural",
    "Thai": "th-TH-PremwadeeNeural",
    "Vietnamese": "vi-VN-HoaiMyNeural",
    "Indonesian": "id-ID-GadisNeural",
    "Malay": "ms-MY-YasminNeural",
    "Czech": "cs-CZ-VlastaNeural",
    "Danish": "da-DK-ChristelNeural",
    "Finnish": "fi-FI-NooraNeural",
    "Hungarian": "hu-HU-NoemiNeural",
    "Norwegian": "nb-NO-PernilleNeural",
    "Romanian": "ro-RO-AlinaNeural",
    "Slovak": "sk-SK-ViktoriaNeural",
    "Swedish": "sv-SE-SofieNeural",
    "Ukrainian": "uk-UA-PolinaNeural"
}

# ============================================
# IMAGE PREPROCESS
# ============================================
def preprocess_image(image):

    img_array = np.array(image)

    gray = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    return binary


# ============================================
# OCR
# ============================================
def extract_text(image, lang):

    custom_config = (
        r'--oem 3 --psm 6'
    )

    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config=custom_config
    )

    return text.strip()


# ============================================
# CLEAN OCR
# ============================================
def clean_text(text):

    prompt = f"""
Fix OCR spelling and formatting.

Return ONLY cleaned text.

TEXT:
{text}
"""

    response = MODEL.generate_content(
        prompt
    )

    return response.text.strip()


# ============================================
# TRANSLATE
# ============================================
def translate_text(
        text,
        target_language
):

    prompt = f"""
Translate text into
{target_language}.

Return ONLY translated text.

TEXT:
{text}
"""

    response = MODEL.generate_content(
        prompt
    )

    return response.text.strip()


# ============================================
# DEEP SENTIMENT + EMOTION
# ============================================
def analyze_emotion(text):

    prompt = f"""
Analyze emotional depth of this text.

Return ONLY valid JSON.

Format:

{{
"sentiment":"positive/negative/neutral",
"emotion":"joy/sadness/grief/anger/fear/calm/excitement/love/etc",
"intensity":"low/medium/high",
"speaking_style":"short sentence"
}}

Text:
{text}
"""

    response = MODEL.generate_content(
        prompt
    )

    try:
        return json.loads(
            response.text
        )
    except:
        return {
            "sentiment": "neutral",
            "emotion": "calm",
            "intensity": "medium",
            "speaking_style":
                "normal voice"
        }


# ============================================
# EMOTIONAL VOICE
# ============================================
async def emotional_tts(
    text,
    voice,
    emotion_data
):

    sentiment = emotion_data.get(
        "sentiment",
        "neutral"
    ).lower()

    emotion = emotion_data.get(
        "emotion",
        "calm"
    ).lower()

    intensity = emotion_data.get(
        "intensity",
        "medium"
    ).lower()

    rate = "+0%"
    pitch = "+0Hz"

    # Sad / grief
    if emotion in [
        "sadness",
        "grief",
        "sorrow"
    ]:
        rate = "-35%"
        pitch = "-15Hz"

    # Joy
    elif emotion in [
        "joy",
        "happiness",
        "excitement"
    ]:
        rate = "+15%"
        pitch = "+10Hz"

    # Anger
    elif emotion == "anger":
        rate = "+8%"
        pitch = "+4Hz"

    # Calm
    elif emotion == "calm":
        rate = "-5%"
        pitch = "-3Hz"

    # Fear
    elif emotion == "fear":
        rate = "-12%"
        pitch = "-8Hz"

    # intensity adjustment
    if intensity == "high":
        rate = rate.replace(
            "%",
            ""
        )

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )

    await communicate.save(
        temp_audio.name
    )

    return temp_audio.name


# ============================================
# PROCESS DOCUMENT
# ============================================
def process_document(
    image,
    source_lang,
    target_lang
):

    processed_img = preprocess_image(
        image
    )

    raw_text = extract_text(
        processed_img,
        OCR_LANGUAGES[
            source_lang
        ]
    )

    if not raw_text:
        st.error(
            "No text detected."
        )
        return

    tab1, tab2, tab3 = st.tabs([
        "Raw OCR",
        "Cleaned Text",
        "Translation + Emotion"
    ])

    # ========================================
    # TAB 1
    # ========================================
    with tab1:
        st.text_area(
            "Raw Text",
            raw_text,
            height=250
        )

    # ========================================
    # TAB 2
    # ========================================
    with tab2:

        with st.spinner(
            "Cleaning OCR..."
        ):

            cleaned = clean_text(
                raw_text
            )

            st.text_area(
                "Cleaned Text",
                cleaned,
                height=250
            )

    # ========================================
    # TAB 3
    # ========================================
    with tab3:

        with st.spinner(
            "Translating..."
        ):

            translated = (
                translate_text(
                    raw_text,
                    target_lang
                )
            )

            st.subheader(
                "Translated Text"
            )

            st.success(
                translated
            )

        st.divider()

        # ORIGINAL EMOTION
        with st.spinner(
            "Understanding original emotion..."
        ):

            original_emotion = (
                analyze_emotion(
                    raw_text
                )
            )

            st.subheader(
                "Original Emotion"
            )

            st.json(
                original_emotion
            )

            voice = TTS_VOICES[
                target_lang
            ]

            original_audio = (
                asyncio.run(
                    emotional_tts(
                        raw_text,
                        voice,
                        original_emotion
                    )
                )
            )

            st.audio(
                original_audio
            )

        st.divider()

        # TRANSLATED EMOTION
        with st.spinner(
            "Understanding translated emotion..."
        ):

            translated_emotion = (
                analyze_emotion(
                    translated
                )
            )

            st.subheader(
                "Translated Emotion"
            )

            st.json(
                translated_emotion
            )

            translated_audio = (
                asyncio.run(
                    emotional_tts(
                        translated,
                        voice,
                        translated_emotion
                    )
                )
            )

            st.audio(
                translated_audio
            )


# ============================================
# UI
# ============================================
st.title(
    "👁️ AudioVision AI"
)

st.write(
    """
OCR + Translation +
Emotion Aware Voice
"""
)

st.sidebar.header(
    "Language Settings"
)

source_lang = (
    st.sidebar.selectbox(
        "OCR Language",
        list(
            OCR_LANGUAGES.keys()
        )
    )
)

target_lang = (
    st.sidebar.selectbox(
        "Translate To",
        list(
            TTS_VOICES.keys()
        )
    )
)

mode = st.radio(
    "Choose Input Method",
    [
        "Upload Image",
        "Live Camera"
    ]
)

# ============================================
# UPLOAD
# ============================================
if mode == "Upload Image":

    uploaded = (
        st.file_uploader(
            "Upload Image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ]
        )
    )

    if uploaded:

        image = Image.open(
            uploaded
        )

        st.image(
            image,
            width=700
        )

        if st.button(
            "Process"
        ):
            process_document(
                image,
                source_lang,
                target_lang
            )

# ============================================
# CAMERA
# ============================================
else:

    camera_img = (
        st.camera_input(
            "Capture Text"
        )
    )

    if camera_img:

        image = Image.open(
            camera_img
        )

        st.image(
            image,
            width=700
        )

        if st.button(
            "Translate Camera Text"
        ):
            process_document(
                image,
                source_lang,
                target_lang
            )
