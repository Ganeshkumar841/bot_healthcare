# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import os
import re
import asyncio
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import faiss
import numpy as np
from PIL import Image
import io

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

# Import custom modules (assuming they exist in your project structure)
# from stt import transcribe_audio
# from translator import detect_and_translate
# from tts import generate_speech_base64

load_dotenv()

# ==============================================================================
# 2. AI Model and RAG Configuration
# ==============================================================================

# --- Firebase Initialization ---
try:
    # This now correctly looks for the 'firebase-credentials.json' file you added.
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Successfully connected to Firestore.")
except Exception as e:
    print(f"Could not connect to Firestore. Place 'firebase-credentials.json' in the root folder. Error: {e}")
    db = None


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
# --- MODIFIED: Use a valid and current generative model ---
GENERATIVE_MODEL = "gemini-2.5-flash-lite"
#need to change to the new model after some time


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
        # --- MODIFIED: Added a stronger, more direct rule about omitting empty sections ---
        SYSTEM_INSTRUCTION = """
        You are ArogyaMitra AI, an expert, friendly, and empathetic voice-based health advisor. Your primary role is to provide safe, helpful, and uniquely structured health information. You must adhere to the following rules at all times.

        --- CORE DIRECTIVES ---
        1.  **NEVER REFUSE A HEALTH QUESTION:** You are designed to provide health guidance. Never state that you are "not a medical expert" or "cannot provide medical advice." Always provide the best possible information based on your training and the provided context, followed by the disclaimer.

        2.  **PERSONALIZE SPARINGLY:** If a user's name is provided, greet them by name once at the start. Avoid repeating their name to keep the conversation natural.

        3.  **ANALYZE USER INTENT & INPUT TYPE (CRITICAL):**
            -   **General Conversation & Greetings:** For greetings (Hello, Hi), thanks, or non-health questions, respond conversationally and briefly. DO NOT use the structured health template.
            -   **Image Analysis:**
                -   If an image has a text prompt, analyze it in context and give the structured health response.
                -   If an image has no text, describe it if it's health-related and ask for context (e.g., "I see what appears to be a skin rash. Can you tell me more about your symptoms?"). If not health-related, ask for a health question related to it.
                -   Never give a full structured response for an image-only query until the user provides text context.
            -   **Health-Related Query:** For any text-based health query, IMMEDIATELY follow the "Structured Health Response" format below.

        4.  **STRUCTURED HEALTH RESPONSE (MANDATORY TEMPLATE):** For any health query, structure your response using the following sections. The output must be clean and natural. **CRITICAL RULE ON EMPTY SECTIONS: If the provided context does not contain information for a specific section (e.g., "Dietary Guidance"), you MUST OMIT THE ENTIRE SECTION, INCLUDING ITS HEADING. DO NOT write "The provided context does not offer any direct information..." or any similar phrase. The section should be completely absent from the response.**

            -   **Empathetic Opening:** Start directly with a positive and reassuring paragraph. Acknowledge their concern. **DO NOT use the heading "Empathetic Opening".** This paragraph should be the very beginning of your response. Example: "I understand that dealing with [symptom] can be worrying, but I'm here to provide some clear information and guidance."

            -   **Primary Precautions:** Use the heading "**Primary Precautions**" and list 2-3 simple, immediate actions.

            -   **Secondary Precautions:** Use the heading "**Secondary Precautions**" and list 2-3 next-level actions or remedies.

            -   **Dietary Guidance:** Use the heading "**Dietary Guidance**" and provide categorized lists for:
                -   Foods to Include (Vegetarian)
                -   Foods to Include (Non-Vegetarian)
                -   Foods to Avoid (Vegetarian)
                -   Foods to Avoid (Non-Vegetarian)

            -   **Peak Stage Symptoms (Warning Signs):** Use the heading "**Peak Stage Symptoms (Warning Signs)**" to list critical symptoms requiring immediate attention.

            -   **When to Consult a Doctor:** Use the heading "**When to Consult a Doctor**". State when to see a doctor and suggest the type of specialist.

            -   **Polite Disclaimer:** End with this exact phrase: "Please remember, this information is for guidance and is not a substitute for professional medical advice from a qualified doctor."

            -   **Engaging Follow-up:** Ask a question to encourage interaction. Example: "Would you like me to elaborate on any of these points?"

        5.  **QUICK REPLIES (MANDATORY for Health Queries):** After the disclaimer, suggest 2-3 relevant follow-up questions. Enclose each in <qr> tags. Example: "<qr>Common Causes</qr><qr>Home Remedies</qr>"

        6.  **DATA GROUNDING:** Base your answers primarily on the "Context from Medical Encyclopedia" provided in the prompt. Synthesize this information into a helpful, easy-to-understand response in the specified format.

        7.  **VOICE-FIRST IDENTITY:** You are a voice-based assistant. Your responses WILL be converted to speech. Never claim you are a 'text-based AI'.
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

def get_health_response(question, language_code="en-US", user_name=None, image_base64=None, chat_history=None):
    if not GENAI_AVAILABLE or not generative_model:
        return "The AI health assistant is currently unavailable. Please check server logs.", []

    language_name = "English" # Hardcoding for now if map is not available
    context_from_book = find_best_chunks(question)
    context_str = "\n".join(context_from_book)

    user_context = f"A user named '{user_name}' is asking." if user_name else "A user is asking."

    full_prompt = f"""
        My entire response MUST be in {language_name}.
        {user_context}

        **CRITICAL INSTRUCTION:** You MUST base your response on the following "Context from Medical Encyclopedia". Analyze it, synthesize it, and present it in the structured format required by your system instructions. Do not invent information. If the context is insufficient, state what you can based on the context and then provide general, safe advice.

        **Context from Medical Encyclopedia:**
        ---
        {context_str if context_str else "No specific context was found in the provided book for this query. Provide a general, safe response based on your training."}
        ---

        **User's query to process:** "{question}"
    """
    
    model_input_parts = [full_prompt]
    if image_base64:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            model_input_parts.append(image)
            print("Successfully prepared image for multimodal input.")
        except Exception as e:
            print(f"Error processing image for model: {e}")
            pass
            
    try:
        formatted_history = []
        if chat_history:
            for message in chat_history:
                role = "user" if message.get("type") == "user" else "model"
                if message.get("content") and "Hello! I'm your AI Health Advisor" not in message.get("content"):
                    formatted_history.append({"role": role, "parts": [{"text": message.get("content")}]})

        chat = generative_model.start_chat(history=formatted_history)
        response = chat.send_message(model_input_parts)
        
        raw_text = response.text
        
        # --- MODIFIED: Added robust post-processing to programmatically remove empty sections ---
        # 1. Initial cleaning of list markers and unwanted headers
        processed_text = re.sub(r'^\s*[A-H]\.\s*', '', raw_text, flags=re.MULTILINE)
        processed_text = processed_text.replace("Empathetic Opening:", "").strip()

        # 2. Define a pattern to find empty sections and remove them
        sections_to_check = [
            "Dietary Guidance",
            "Peak Stage Symptoms \(Warning Signs\)", # Escape parentheses for regex
            "When to Consult a Doctor"
        ]
        no_info_phrase = "The provided context does not offer any direct information on this topic"
        
        for section in sections_to_check:
            # This regex finds a section heading (bolded or not), followed by the "no info" phrase, and removes the entire block.
            pattern = re.compile(rf"(\*\*|){section}(\*\*|)\s*{re.escape(no_info_phrase)}\.?\s*\n?", re.IGNORECASE)
            processed_text = pattern.sub("", processed_text)

        # 3. Final cleanup: Extract quick replies and clean the final text
        quick_replies = re.findall(r'<qr>(.*?)</qr>', processed_text)
        clean_text = re.sub(r'<qr>.*?</qr>', '', processed_text).strip()

        return clean_text, quick_replies

    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return "An error occurred while trying to get a response from the AI model.", []

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
        user_question = data.get("question", "")
        language = data.get("language", "en-US")
        
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"error": "User ID is missing. Authentication may have failed."}), 400

        user_name = data.get("userName")
        image_base64 = data.get("imageBase64")
        chat_history = data.get("chatHistory", [])

        if image_base64:
            # Clean up the base64 prefix
            image_base64 = re.sub('^data:image/.+;base64,', '', image_base64)

        if not user_question and not image_base64:
            return jsonify({"error": "No question or image provided"}), 400

        # Create a placeholder question if only an image is provided
        if not user_question and image_base64:
            question_for_model = "The user has not provided any text. Please analyze the image provided according to the system instructions for image-only analysis."
        else:
            question_for_model = user_question

        # Get the AI's response
        answer, quick_replies = get_health_response(
            question_for_model, 
            language, 
            user_name, 
            image_base64, 
            chat_history 
        )

        audio_base64 = ""
        # TTS logic can be added back here if needed.

        return jsonify({"answer": answer, "audio": audio_base64, "quick_replies": quick_replies})
    except Exception as e:
        print(f"A critical error occurred in the /ask route: {e}")
        return jsonify({"error": "A critical server error occurred. Please check the backend logs for details."}), 500

if __name__ == "__main__":
    app.run(debug=True)

