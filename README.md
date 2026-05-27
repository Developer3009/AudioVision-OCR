# 👁️ AudioVision AI

AI-powered OCR, Translation, Emotion Analysis, and Emotion-Aware Voice Generation using Streamlit, Gemini AI, Tesseract OCR, and Edge TTS.

## Features

- 📷 OCR from uploaded images
- 📸 Live camera text capture
- 🌍 Multi-language translation
- 🧠 Emotion and sentiment analysis
- 🔊 Emotion-aware voice output
- 🎭 Voice changes based on emotional depth
- ⚡ Streamlit UI

---

## Tech Stack

- Python
- Streamlit
- OpenCV
- Tesseract OCR
- Gemini API
- Edge TTS
- NumPy
- Pillow

---

## Project Structure

```text
Project/
│── app.py
│── requirements.txt
│── README.md
│── .env.example
│── .gitignore
```

---

## Installation

### 1. Clone repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Create virtual environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
.\venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Tesseract OCR

Download and install:

https://github.com/UB-Mannheim/tesseract/wiki

Default path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

### 5. Configure API key

Create `.env`

```env
GEMINI_API_KEY=your_api_key_here
```

---

### 6. Run app

```bash
streamlit run app.py
```

---

## Supported Languages

- English
- Hindi
- Spanish
- French
- German

---

## Screenshots

Add screenshots here.

---

## Future Improvements

- Real-time live translation
- More languages
- Better emotional voice realism
- Offline mode
- PDF support

---

## License

MIT License