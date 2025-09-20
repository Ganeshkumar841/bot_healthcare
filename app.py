# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import os
import re
import asyncio
import tempfile
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import faiss
import numpy as np

# Import custom modules
from stt import transcribe_audio
from translator import detect_and_translate
from tts import generate_speech_base64

load_dotenv()

# ==============================================================================
# 2. AI Model and RAG Configuration
# ==============================================================================
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Error: The 'google-generativeai' library is not installed.")

API_KEY = os.getenv("GEMINI_API_KEY")

if GENAI_AVAILABLE and API_KEY and API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=API_KEY)
    print("Successfully configured Generative AI.")
else:
    GENAI_AVAILABLE = False
    print("Warning: GENAI_AVAILABLE is False. Check if API key is set in .env")

# --- Model Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-2.0-flash-lite"

# --- FFmpeg Path (for Vosk audio conversion) ---
ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")
if os.name == 'nt':
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# --- FAISS Index (RAG) Loading ---
FAISS_INDEX_PATH = "health_book.index"
TEXT_CHUNKS_PATH = "health_book_chunks.txt"
faiss_index = None
text_chunks = []
try:
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            text_chunks = f.read().split("\n---\n")
        print("Successfully loaded FAISS index and text chunks.")
    else:
        print("Warning: RAG files not found. Bot will run without book context.")
except Exception as e:
    print(f"Error loading RAG files: {e}")
    faiss_index = None

# --- Generative Model Initialization ---
generative_model = None
if GENAI_AVAILABLE:
    try:
        SYSTEM_INSTRUCTION = """
        You are an AI Health Advisor. Your persona is supportive, knowledgeable, and cautious. Your primary goal is to provide helpful, safe, and informative content based on the provided context from a medical encyclopedia. Your tone must be concise, clear, and supportive.

        **First, analyze the user's query to understand its intent. Classify it into one of the following categories:**
        1.  **Specific Ailment/Suffering:** The user is describing symptoms or a health problem (e.g., "I have a headache," "my friend has back pain").
        2.  **Fitness/Exercise Related:** The query is about exercises, workout injuries, or fitness concepts (e.g., "how to do a deadlift?").
        3.  **General Health Awareness:** The query is about a disease, condition, or general wellness (e.g., "what is hypertension?").
        4.  **Diet and Nutrition:** The query is about food, diets, or nutritional advice.
        5.  **Vague Follow-up:** The user gives a short, non-specific reply like "ok," or "tell me more."
        6.  **Simple Greeting:** The user is asking a non-medical, simple greeting like "hi", "hello", "namaste".

        **Based on the classification, tailor your response structure as follows:**

        ---
        **For a 'Simple Greeting' query:**
        * Provide a short, friendly, non-medical response like: "Hello! How can I help you with your health questions today?"
        * DO NOT provide any disclaimers or medical information.
        ---
        **For a 'Specific Ailment/Suffering' query:**
        1.  **Mandatory Disclaimer:** Start with: "**Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns.**"
        2.  **Direct Answer:** Briefly explain possible causes based on the query and context.
        3.  **Immediate Care/First Aid (if applicable):** Provide simple, safe steps for immediate relief.
        4.  **When to Consult a Doctor:** State specific "red flag" symptoms for when to see a doctor immediately.
        5.  **Key Precautions:** List 2-3 important "do's and don'ts".
        6.  **Closing:** Ask a supportive follow-up question.
        ---
        **For other medical queries (Fitness, General, Nutrition):**
        1.  **Educational Disclaimer:** Start with: "**Disclaimer: This information is for educational purposes. Always consult with a qualified healthcare provider for personalized medical or dietary advice.**"
        2.  **Comprehensive Answer:** Provide a clear, detailed explanation of the topic.
        3.  **Key Takeaways:** Summarize the most important points.
        4.  **Closing:** Ask if the user would like to dive deeper into any specific aspect.
        ---
        **For a 'Vague Follow-up' query:**
        * Ask for clarification based on the conversational history. Example: "Which point would you like me to elaborate on?".
        """
        generative_model = genai.GenerativeModel(
            GENERATIVE_MODEL,
            system_instruction=SYSTEM_INSTRUCTION
        )
        print("Generative model initialized successfully.")
    except Exception as e:
        print(f"Could not initialize generative model: {e}")
        GENAI_AVAILABLE = False

# ==============================================================================
# 3. Helper Functions
# ==============================================================================
VOICE_MAP = {
    "en": "en-US-AriaNeural", "es": "es-MX-DaliaNeural", "hi": "hi-IN-SwaraNeural",
    "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural", "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural", "or": "or-IN-NiranjanNeural"
}
LANGUAGE_MAP = { "en": "English", "es": "Spanish", "hi": "Hindi", "fr": "French", "de": "German", "te": "Telugu", "ta": "Tamil", "or": "Odia" }
VACCINATION_SCHEDULE = {
    "en": { "title": "National Immunization Schedule (India)", "schedule": { "Birth": "BCG, Oral Polio Vaccine (OPV 0), Hepatitis B (Birth dose)", "6 Weeks": "OPV 1, DPT 1, Hepatitis B 1, Rotavirus 1, PCV 1", "10 Weeks": "OPV 2, DPT 2, Hepatitis B 2, Rotavirus 2, PCV 2", "14 Weeks": "OPV 3, DPT 3, Hepatitis B 3, Rotavirus 3, PCV 3", "9-12 Months": "Measles & Rubella (MR) 1st Dose, PCV Booster", "16-24 Months": "DPT Booster 1, OPV Booster, MR 2nd Dose", "5-6 Years": "DPT Booster 2", "10 Years": "Tetanus and adult Diphtheria (Td) vaccine", "16 Years": "Tetanus and adult Diphtheria (Td) vaccine", "Pregnant Women": "Two doses of Td vaccine, and one Td booster if previously vaccinated." }, "details": { "BCG": "Bacillus Calmette-Guérin, for Tuberculosis (TB) protection.", "OPV": "Oral Polio Vaccine, protects against Polio.", "Hepatitis B": "Protects against Hepatitis B virus infection.", "DPT": "Diphtheria, Pertussis (Whooping Cough), and Tetanus.", "Rotavirus": "Protects against Rotavirus diarrhea.", "PCV": "Pneumococcal Conjugate Vaccine, protects against certain types of pneumonia and meningitis.", "MR": "Measles and Rubella vaccine.", "Td": "Tetanus and adult Diphtheria vaccine." } },
    "hi": { "title": "राष्ट्रीय टीकाकरण सारणी (भारत)", "schedule": { "जन्म": "बीसीजी, ओरल पोलियो वैक्सीन (ओपीवी 0), हेपेटाइटिस बी (जन्म खुराक)", "6 सप्ताह": "ओपीवी 1, डीपीटी 1, हेपेटाइटिस बी 1, रोटावायरस 1, पीसीवी 1", "10 सप्ताह": "ओपीवी 2, डीपीटी 2, हेपेटाइटिस बी 2, रोटावायरस 2, पीसीवी 2", "14 सप्ताह": "ओपीवी 3, डीपीटी 3, हेपेटाइटिस बी 3, रोटावायरस 3, पीसीवी 3", "9-12 महीने": "खसरा और रूबेला (एमआर) पहली खुराक, पीसीवी बूस्टर", "16-24 महीने": "डीपीटी बूस्टर 1, ओपीवी बूस्टर, एमआर दूसरी खुराक", "5-6 साल": "डीपीटी बूस्टर 2", "10 साल": "टेटनस और वयस्क डिप्थीरिया (टीडी) वैक्सीन", "16 साल": "टेटनस और वयस्क डिप्theria (टीडी) वैक्सीन", "गर्भवती महिलाएं": "टीडी वैक्सीन की दो खुराक, और पहले टीका लगने पर एक टीडी बूस्टर।" }, "details": { "BCG": "बैसिलस कैलमेट-गुएरिन, तपेदिक (टीबी) से सुरक्षा के लिए।", "ओपीवी": "ओरल पोलियो वैक्सीन, पोलियो से बचाता है।", "हेपेटाइटिस बी": "हेपेटाइटिस बी वायरस संक्रमण से बचाता है।", "डीपीटी": "डिप्थीtheria, पर्टुसिस (काली खांसी), और टेटनस।", "रोटावायरस": "रोटावायरस दस्त से बचाता है।", "पीसीवी": "न्यूमोकोकल कंजुगेट वैक्सीन, कुछ प्रकार के निमोनिया और मेनिनजाइटिस से बचाता है।", "एमआर": "खसरा और रूबेला वैक्सीन।", "टीडी": "टेटनस और वयस्क डिप्थीरिया वैक्सीन।" } }
}
VACCINE_KEYWORDS = ['vaccine', 'vaccination', 'immunization', 'schedule', 'dpt', 'opv', 'bcg', 'polio', 'measles', 'rubella', 'hepatitis', 'tetanus', 'rota', 'pcv', 'mr', 'td', 'टीका', 'टीकाकरण']

def is_vaccine_question(question):
    return any(keyword in question.lower() for keyword in VACCINE_KEYWORDS)

def get_vaccine_response(language_code="en"):
    lang_data = VACCINATION_SCHEDULE.get(language_code, VACCINATION_SCHEDULE["en"])
    schedule_info, details_info = lang_data['schedule'], lang_data['details']
    response = f"**{lang_data['title']}**\n\n"
    for age, vaccines in schedule_info.items():
        response += f"- **{age}:** {vaccines}\n"
    response += "\n**Vaccine Details:**\n"
    for vaccine, detail in details_info.items():
        response += f"- **{vaccine}:** {detail}\n"
    return response

def find_best_chunks(question, top_k=3):
    if not faiss_index or not text_chunks or not GENAI_AVAILABLE: return []
    try:
        query_embedding_result = genai.embed_content(model=EMBEDDING_MODEL, content=question, task_type="RETRIEVAL_QUERY")
        query_embedding = np.array(query_embedding_result['embedding']).astype('float32').reshape(1, -1)
        _, indices = faiss_index.search(query_embedding, top_k)
        return [text_chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

def get_health_response(question, language_code="en", history=[]):
    if not GENAI_AVAILABLE or not generative_model:
        return "The AI health assistant is currently unavailable. Please check server logs."
    if is_vaccine_question(question):
        return get_vaccine_response(language_code)
    language_name = LANGUAGE_MAP.get(language_code, "English")
    context_from_book = find_best_chunks(question)
    context_str = "\n".join(context_from_book)
    full_prompt = f"""
        My entire response MUST be in {language_name}.
        **Context from Medical Encyclopedia for the CURRENT question:**
        ---
        {context_str if context_str else "No specific context was found in the book for this query. Answer based on your general knowledge but be extremely cautious and prioritize safety."}
        ---
        **User's new question:** "{question}"
    """
    try:
        chat = generative_model.start_chat(history=history)
        response = chat.send_message(full_prompt)
        return response.text if hasattr(response, 'text') and response.text else "I couldn't generate a response."
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return "An error occurred while trying to get a response from the AI model."

# ==============================================================================
# 4. Flask Application and Routes
# ==============================================================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")
    language = data.get("language", "en")
    history = data.get("history", [])
    source = data.get("source", "text")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    answer = get_health_response(user_question, language, history)
    audio_base64 = ""
    if source == 'voice':
        voice = VOICE_MAP.get(language)
        if voice:
            try:
                speech_text = re.sub(r'[\*#]', '', answer)
                audio_base64 = asyncio.run(generate_speech_base64(speech_text, voice=voice))
            except Exception as e:
                print(f"Error during TTS generation for language {language}: {e}")
    return jsonify({"answer": answer, "audio": audio_base64})

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio_data']
    try:
        # Securely create an isolated temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, 'temp_audio.webm')
            audio_file.save(temp_path)
            # Call the Vosk transcription function
            transcribed_text = transcribe_audio(temp_path)
            # Translate the result to English for the model
            final_text = detect_and_translate(transcribed_text, target_lang='en')
        return jsonify({"transcription": final_text})
    except Exception as e:
        print(f"Error during transcription/translation: {e}")
        return jsonify({"error": "Failed to process audio"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

