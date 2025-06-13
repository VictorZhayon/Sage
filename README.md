# 🧙🏿‍♂️ Sage AI Assistant

Sage AI Assistant is a mystical document question-answering app powered by Gemini AI. Present your manuscripts (PDF, TXT, DOCX) and seek wisdom from their pages! The Sage will contemplate your queries and provide answers, citing the relevant passages.

## Features

- 📁 Upload manuscripts (PDF, TXT, DOCX)
- 🧠 Documents are chunked and stored for efficient retrieval
- 💬 Ask questions and receive answers grounded in your manuscripts
- 🔍 Sources for each answer are revealed for transparency
- 🪄 Sage-inspired UI and responses

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VictorZhayon/sage-ai-assistant.git
cd sage-ai-assistant
```

### 2. Install Requirements

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Set Up Gemini API Key

- Obtain your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- You can set it as an environment variable or enter it in the app sidebar.

### 4. Run the App Locally

```bash
streamlit run app.py
```

## Project Structure

```
sage-ai-assistant/
│
├── app.py                  # Main Streamlit app
├── config.py               # Configuration settings
├── document_processor.py   # Document processing logic
├── vector_store.py         # Vector database logic
├── gemini_client.py        # Gemini API client
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Usage

1. **Offer Manuscripts:** Upload your documents to the Sage.
2. **Seek Wisdom:** Ask questions and receive answers based on your manuscripts.
3. **Review Sources:** Expand the sources to see the passages the Sage consulted.

## License

MIT License

---

May the Sage illuminate your path to knowledge!