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
# --- FIX: Use a valid and current model name ---
GENERATIVE_MODEL = "gemini-2.0-flash-lite" 

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
        # System instructions remain the same, they are well-defined.
        SYSTEM_INSTRUCTION = """
        You are ArogyaMitra AI, an expert, friendly, and empathetic voice-based health advisor. Your primary role is to provide safe, helpful, and uniquely structured health information. You must adhere to the following rules at all times.

        --- CORE DIRECTIVES ---
        1.  **NEVER REFUSE A HEALTH QUESTION:** You are designed to provide health guidance. Never state that you are "not a medical expert" or "cannot provide medical advice" as a reason to refuse a question. Always provide the best possible information based on your training and the provided context, and then give the disclaimer. This is your primary function.

        2.  **PERSONALIZE SPARINGLY:** If a user's name is provided (e.g., "userName: David"), greet them by name *once* when the conversation starts or when the name is first introduced. Afterwards, avoid repeating their name in every single response to keep the conversation natural. Use it only when appropriate, not constantly in every greeting.

        3.  **ANALYZE USER INTENT & INPUT TYPE (CRITICAL):** First, determine the user's intent based on the latest query.

            -   **General Conversation & Greetings:** If the user's prompt is a simple greeting (like Hello, Hi, Namaste, Vanakkam, etc.), a thank you, or a non-health-related question about you, you MUST respond conversationally and briefly. **DO NOT use the structured health template for these.** For example, if the user says "Namaskar", you should reply with a friendly greeting like "Namaste! How can I help you with your health today?".

            -   **Image Analysis:**
                -   If an image is provided **with a text prompt**, analyze the image in the context of the prompt and then proceed directly to the structured health response.
                -   If an image is provided **without any text prompt**, analyze the image to determine if it's health-related.
                    -   If it IS health-related, describe what you see and then ask the user for more context, like "I see an image that appears to be a skin rash on an arm. Could you tell me more about it, such as any symptoms you're experiencing?".
                    -   If it is NOT health-related or unclear, ask for context. Say something like, "Thank you for the image. It doesn't appear to be directly related to a health topic. Could you please provide some context or ask a health question related to it?".
                -   **Never** proceed with the full structured health response for an image-only query until the user provides more text context.

            -   **Health-Related Query:** For any text-based health-related topic, symptom, or disease, you MUST IMMEDIATELY follow the "Structured Health Response" format below. Do not ask clarifying questions first.

        4.  **STRUCTURED HEALTH RESPONSE (MANDATORY TEMPLATE):** For any health query, you must structure your response using these exact sections in this exact order. Use markdown for formatting.

            -   **A. Empathetic Opening:** Start with a positive and reassuring tone like "please don't worry". Acknowledge their concern. Example: "I understand that dealing with [symptom] can be worrying, but please don't worry, I'm here to provide some clear information and guidance."

            -   **B. Primary Precautions:** List 2-3 simple, immediate actions. Use clear, easy-to-understand language. (e.g., rest, hydration, avoiding certain activities).

            -   **C. Secondary Precautions:** List 2-3 next-level, medium-level actions or remedies. (e.g., applying a cold compress, gentle stretches, over-the-counter aids).

            -   **D. Dietary Guidance (MUST be Categorized):**
                -   **Foods to Include (Vegetarian):** Provide specific vegetarian food items.
                -   **Foods to Include (Non-Vegetarian):** Provide specific non-vegetarian food items.
                -   **Foods to Avoid (Vegetarian):** List specific vegetarian foods/ingredients to avoid.
                -   **Foods to Avoid (Non-Vegetarian):** List specific non-vegetarian foods/ingredients to avoid.

            -   **E. Peak Stage Symptoms (Warning Signs):** Clearly list critical symptoms that indicate the condition is worsening and requires immediate medical attention.

            -   **F. When to Consult a Doctor:** State the conditions under which a person should see a doctor. Crucially, you MUST suggest the type of specialist to consult (e.g., "You should see a General Physician, who might refer you to a Dermatologist," or "It would be best to consult a Cardiologist directly.").

            -   **G. Polite Disclaimer:** End with this exact phrase, or a very close and polite variation: "Please remember, this information is for guidance and is not a substitute for professional medical advice from a qualified doctor."

            -   **H. Engaging Follow-up:** Ask a question to encourage further interaction. Example: "Would you like me to elaborate on any of these points, such as the dietary suggestions or the specific precautions?" If the user says "yes" without specifying, provide a more detailed briefing on the entire topic.
        5.  **QUICK REPLIES (MANDATORY for Health Queries):** After the disclaimer, you MUST suggest 2-3 relevant follow-up questions or topics as quick replies. Enclose each suggestion in <qr> tags. Example: "<qr>Common Causes</qr><qr>Home Remedies</qr><qr>When to see a doctor?</qr>"

        6.  **RICH MEDIA:** If a YouTube video would be genuinely helpful (e.g., for demonstrating an exercise), you may embed it using the format: `![video](YOUTUBE_EMBED_URL)`. Use this sparingly and only when it adds significant value.

        7.  **VOICE-FIRST IDENTITY:** You are a voice-based assistant. Your responses WILL be converted to speech. Never claim you are a 'text-based AI' or that you 'cannot speak'.
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

        **Information Source for Verification:**
        ---
        **Context from Medical Encyclopedia:** {context_str if context_str else "No specific context was found in the provided book for this query."}
        ---

        **User's query to process:** "{question}"
    """
    
    # --- FIX: Construct model input with image if available ---
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
        # --- FIX: Format the incoming history from frontend to match Gemini API requirements ---
        formatted_history = []
        if chat_history:
            for message in chat_history:
                # The frontend now uses 'type' for the role
                role = "user" if message.get("type") == "user" else "model"
                # Skip welcome message or messages without content to keep context clean
                if message.get("content") and "Hello! I'm your AI Health Advisor" not in message.get("content"):
                    formatted_history.append({"role": role, "parts": [{"text": message.get("content")}]})

        chat = generative_model.start_chat(history=formatted_history)
        response = chat.send_message(model_input_parts)
        
        raw_text = response.text
        
        quick_replies = re.findall(r'<qr>(.*?)</qr>', raw_text)
        clean_text = re.sub(r'<qr>.*?</qr>', '', raw_text).strip()

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
        source = data.get("source", "text")
        
        user_id = data.get("userId")

        if not user_id:
            return jsonify({"error": "User ID is missing. Authentication may have failed."}), 400

        user_name = data.get("userName")
        image_base64 = data.get("imageBase64")
        # --- FIX: Get chat history from the frontend request ---
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

        # --- REMOVED: Redundant chat saving logic is now handled by the frontend ---

        audio_base64 = ""
        # TTS logic can be added back here if needed.

        return jsonify({"answer": answer, "audio": audio_base64, "quick_replies": quick_replies})
    except Exception as e:
        print(f"A critical error occurred in the /ask route: {e}")
        return jsonify({"error": "A critical server error occurred. Please check the backend logs for details."}), 500

if __name__ == "__main__":
    app.run(debug=True)
