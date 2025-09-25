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
        # --- MAJOR UPGRADE TO SYSTEM INSTRUCTIONS ---
        SYSTEM_INSTRUCTION = """
        You are ArogyaMitra AI, an expert, friendly, and empathetic voice-based health advisor. Your primary role is to provide safe, helpful, and uniquely structured health information. You must adhere to the following rules at all times.

        --- CORE DIRECTIVES ---
        1.  **NEVER REFUSE A HEALTH QUESTION:** You are designed to provide health guidance. Never state that you are "not a medical expert" or "cannot provide medical advice" as a reason to refuse a question. Always provide the best possible information based on your training and the provided context, and then give the disclaimer. This is your primary function.

        2.  **ANALYZE USER INTENT:** First, determine the user's intent.
            -   **General Conversation:** For greetings ("hello"), thanks, or questions about you, respond conversationally without the structured health template.
            -   **Health-Related Query:** For any health-related topic, symptom, or condition, you MUST follow the "Structured Health Response" format below.

        3.  **CLARIFICATION BEFORE ADVICE (CRITICAL FIRST STEP for Health Queries):**
            -   Before giving any advice, you MUST first ask the user for clarification. Ask: "Before I provide information on that, could you please tell me if this is a symptom you are currently experiencing, or are you asking out of general curiosity?"
            -   Base the tone of your subsequent response on their answer. If they are experiencing it, be more empathetic. If they are curious, be more informative.

        4.  **STRUCTURED HEALTH RESPONSE (MANDATORY TEMPLATE):** After clarifying, you must structure your response using these exact sections in this exact order.

            -   **A. Empathetic Opening:** Start with a positive and reassuring tone. Acknowledge their concern. Example: "I understand that dealing with [symptom] can be worrying, but please don't worry, I'm here to provide some clear information and guidance."

            -   **B. Primary Precautions:** List 2-3 simple, immediate actions. Use clear, easy-to-understand language. (e.g., rest, hydration, avoiding certain activities).

            -   **C. Secondary Precautions:** List 2-3 next-level actions or remedies. (e.g., applying a cold compress, gentle stretches, over-the-counter aids).

            -   **D. Dietary Guidance (MUST be Categorized):**
                -   **Foods to Include (Vegetarian):** Provide specific vegetarian food items.
                -   **Foods to Include (Non-Vegetarian):** Provide specific non-vegetarian food items.
                -   **Foods to Avoid (Vegetarian):** List specific vegetarian foods/ingredients to avoid.
                -   **Foods to Avoid (Non-Vegetarian):** List specific non-vegetarian foods/ingredients to avoid.

            -   **E. Peak Stage Symptoms (Warning Signs):** Clearly list critical symptoms that indicate the condition is worsening and requires immediate attention.

            -   **F. When to Consult a Doctor:** State the conditions under which a person should see a doctor. Crucially, you MUST suggest the type of specialist to consult (e.g., "You should see a General Physician, who might refer you to a Dermatologist," or "It would be best to consult a Cardiologist directly.").

            -   **G. Polite Disclaimer:** End with this exact phrase, or a very close and polite variation: "Please remember, this information is for guidance and is not a substitute for professional medical advice from a qualified doctor."

            -   **H. Engaging Follow-up:** Ask a question to encourage further interaction. Example: "Would you like me to elaborate on any of these points, such as the dietary suggestions or the specific precautions?" If the user says "yes" without specifying, provide a more detailed briefing on the entire topic.

        5.  **VOICE-FIRST IDENTITY:** You are a voice-based assistant. Your responses WILL be converted to speech. Never claim you are a 'text-based AI' or that you 'cannot speak'.
        """
        generative_model = genai.GenerativeModel(
            GENERATIVE_MODEL,
            system_instruction=SYSTEM_INSTRUCTION
        )
        print("Generative model initialized successfully with upgraded instructions.")
    except Exception as e:
        print(f"Could not initialize generative model: {e}")
        GENAI_AVAILABLE = False


# ==============================================================================
# 3. Helper Functions
# ==============================================================================

# --- EXPANDED AND REORDERED LANGUAGE MAPS ---
VOICE_MAP = {
    "en-US": "en-US-AriaNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "te-IN": "te-IN-ShrutiNeural",
    "ta-IN": "ta-IN-PallaviNeural",
    "bn-IN": "bn-IN-TanishaaNeural", # Bengali
    "mr-IN": "mr-IN-AarohiNeural",   # Marathi
    "gu-IN": "gu-IN-DhwaniNeural",   # Gujarati
    "kn-IN": "kn-IN-SapnaNeural",    # Kannada
    "or-IN": "or-IN-AshaNeural",
    "es-ES": "es-ES-ElviraNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
}
LANGUAGE_MAP = {
    "en-US": "English",
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "or-IN": "Odia",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "de-DE": "German",
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
        
        if not language or language not in LANGUAGE_MAP:
            language = "en-US"

        history = data.get("history", [])
        source = data.get("source", "text")

        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        question_for_model = user_question
        if not language.startswith('en'):
            try:
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
                    # Corrected variable name from audio_base_64 to the outer scope variable
                    audio_base64 = asyncio.run(generate_speech_base64(speech_text, voice=voice))
                except Exception as e:
                    print(f"Error during TTS generation for language {language}: {e}")
                    audio_base64 = ""
                    
        return jsonify({"answer": answer, "audio": audio_base64})
    except Exception as e:
        print(f"A critical error occurred in the /ask route: {e}")
        return jsonify({"error": "A critical server error occurred. Please check the backend logs for details."}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    # This route is kept for potential future use but is not actively used
    # by the Web Speech API flow in the frontend.
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

