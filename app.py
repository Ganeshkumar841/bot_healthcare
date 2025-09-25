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

if GENAI_AVAILABLE and API_KEY:
    genai.configure(api_key=API_KEY)
    print("Successfully configured Generative AI.")
else:
    GENAI_AVAILABLE = False
    print("Warning: GENAI_AVAILABLE is False. Check if the GEMINI_API_KEY is set in .env")

# --- Model Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash-latest"

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
        You are ArogyaMitra AI, a friendly and empathetic voice-based health advisor. Your primary role is to provide safe, helpful, and informative content based on verified health knowledge. You must adhere to the following rules at all times:

        --- CORE BEHAVIOR RULES ---
        1.  **Analyze User Intent First:** Before responding, determine if the user is asking a specific health question or having a general conversation (e.g., saying "hello", "thank you", or asking about you).

        2.  **For General Conversation:** If the user is not asking a health question, respond naturally and conversationally. Do NOT use the structured health template.
            -   **CRITICAL RULE: You are a voice-based assistant. Your responses WILL be converted to speech. Never, under any circumstances, claim that you are a 'text-based AI' or that you 'cannot speak'. This is factually incorrect for the system you are in. If a user asks you to speak or repeat something, simply respond naturally, for example by saying "Of course, I can certainly do that." or "Here is the information again:" and then repeat the previous health advice if it is relevant.**

        3.  **For Health Questions:** If the user asks a health question, you MUST use the following structured format. This is not optional.
            -   **Empathetic Opening:** Start with a reassuring and positive tone. Example: "I understand that you're concerned about [symptom], and I'm here to help guide you. Let's go through this together."
            -   **Tiered Precautions:**
                -   **Primary Precautions:** Give 2-3 simple, immediate actions (e.g., rest, hydration).
                -   **Secondary Precautions:** Suggest 2-3 next-level actions (e.g., applying a compress, gentle stretches).
            -   **Dietary Guidance (Categorized):**
                -   **Foods to Include:** Separate suggestions for 'Vegetarian' and 'Non-Vegetarian'.
                -   **Foods to Avoid:** Separate suggestions for 'Vegetarian' and 'Non-Vegetarian'.
            -   **Warning Signs (Peak Stage Symptoms):** List critical symptoms that require immediate attention.
            -   **When to See a Doctor:** Clearly state when professional medical help is necessary and suggest the type of specialist to consult (e.g., cardiologist, neurologist, primary care physician).
            -   **Polite Disclaimer:** End with: "Please remember, I am an AI assistant and this information is not a substitute for professional medical advice."
            -   **Engaging Follow-up Question:** Ask a question to continue the conversation. Example: "Would you like me to elaborate on any of these points, such as the dietary suggestions or the precautions?"

        4.  **Handle Ambiguity and Errors:**
            -   If a user's query is misspelled, incomplete, or ambiguous (e.g., the Telugu word 'kallu' meaning 'eyes' or 'legs'), you MUST ask for clarification before providing a health response. Do not guess. Example: "I see you mentioned 'kallu.' In Telugu, that can mean either 'eyes' or 'legs.' Could you please clarify which one you are referring to so I can provide the most accurate information?"

        5.  **Information Verification:**
            -   Your primary source of information is the context provided from the RAG system (the medical book). State when your information comes from this source.
            -   If no context is found in the book, you may use your general knowledge but you MUST state that the information is from your general training and should be verified with a healthcare professional.
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

# --- FIX: Updated keys to match the frontend values (e.g., 'en-US') ---
VOICE_MAP = {
    "en-US": "en-US-AriaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "ta-IN": "ta-IN-PallaviNeural",
    "or-IN": "or-IN-AshaNeural"
}
LANGUAGE_MAP = {
    "en-US": "English",
    "es-ES": "Spanish",
    "hi-IN": "Hindi",
    "fr-FR": "French",
    "de-DE": "German",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "or-IN": "Odia"
}

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

def get_health_response(question, language_code="en-US", history=[]):
    if not GENAI_AVAILABLE or not generative_model:
        return "The AI health assistant is currently unavailable. Please check server logs."
    
    language_name = LANGUAGE_MAP.get(language_code, "English")
    context_from_book = find_best_chunks(question)
    context_str = "\n".join(context_from_book)

    full_prompt = f"""
        My entire response MUST be in {language_name}.

        **Information Source for Verification:**
        ---
        **Context from Medical Encyclopedia:** {context_str if context_str else "No specific context was found in the provided book for this query."}
        ---

        **User's query to process:** "{question}"
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
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a_super_secret_key")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_question = data.get("question")
        language = data.get("language", "en-US")
        
        # --- FIX: Add a check for an empty or invalid language string ---
        if not language or language not in LANGUAGE_MAP:
            language = "en-US" # Default to English if the language is missing or not supported

        history = data.get("history", [])
        source = data.get("source", "text")

        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        # --- Translate non-English questions to English for the model ---
        question_for_model = user_question
        # FIX: More robust check to avoid translating English to English
        if not language.startswith('en'):
            try:
                # Use the translator utility to convert the user's question to English
                question_for_model = detect_and_translate(user_question, target_lang='en')
                print(f"Translated question from '{language}' to 'en': '{question_for_model}'")
            except Exception as e:
                print(f"Could not translate question, using original. Error: {e}")
                question_for_model = user_question
        
        answer = get_health_response(question_for_model, language, history)
        
        audio_base64 = ""
        if source == 'voice':
            voice = VOICE_MAP.get(language)
            if voice:
                try:
                    speech_text = re.sub(r'[\*#]', '', answer)
                    audio_base_64 = asyncio.run(generate_speech_base64(speech_text, voice=voice))
                except Exception as e:
                    print(f"Error during TTS generation for language {language}: {e}")
                    audio_base_64 = ""
                    
        return jsonify({"answer": answer, "audio": audio_base_64})
    except Exception as e:
        print(f"A critical error occurred in the /ask route: {e}")
        return jsonify({"error": "A critical server error occurred. Please check the backend logs for details."}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio_data']
    language = request.form.get('language', 'en')
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, 'temp_audio.webm')
            audio_file.save(temp_path)
            
            transcribed_text = transcribe_audio(temp_path, lang_code=language)
            final_text = detect_and_translate(transcribed_text, target_lang='en')

            return jsonify({"transcription": final_text})

    except Exception as e:
        print(f"Error during transcription process: {e}")
        return jsonify({"error": f"Failed to process audio. Details: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

