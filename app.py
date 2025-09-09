from flask import Flask, render_template, request, jsonify
import os
import faiss
import numpy as np
import re

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- Configuration ---
# IMPORTANT: It's better practice to use environment variables for API keys.
# I'm leaving your hardcoded key here as it was in your original file, but be aware of this.
API_KEY =  "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4"
if GENAI_AVAILABLE and API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=API_KEY)
else:
    GENAI_AVAILABLE = False
    print("Warning: API_KEY is not set or google.generativeai is not installed.")

# Models
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash"

# File paths
FAISS_INDEX_PATH = "health_book.index"
TEXT_CHUNKS_PATH = "health_book_chunks.txt"

# --- Load RAG Data (Book Index and Chunks) ---
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

# --- Specialized Knowledge & Language ---
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "te": "Telugu",
    "ta": "Tamil",
    "or": "Odia"
}

VACCINATION_SCHEDULE = {
    "title": "National Immunization Schedule (India)",
    "schedule": {
        "Birth": "BCG, Oral Polio Vaccine (OPV 0), Hepatitis B (Birth dose)",
        "6 Weeks": "OPV 1, DPT 1, Hepatitis B 1, Rotavirus 1, PCV 1",
        "10 Weeks": "OPV 2, DPT 2, Hepatitis B 2, Rotavirus 2, PCV 2",
        "14 Weeks": "OPV 3, DPT 3, Hepatitis B 3, Rotavirus 3, PCV 3",
        "9-12 Months": "Measles & Rubella (MR) 1st Dose, PCV Booster",
        "16-24 Months": "DPT Booster 1, OPV Booster, MR 2nd Dose",
        "5-6 Years": "DPT Booster 2",
        "10 Years": "Tetanus and adult Diphtheria (Td) vaccine",
        "16 Years": "Tetanus and adult Diphtheria (Td) vaccine",
        "Pregnant Women": "Two doses of Td vaccine, and one Td booster if previously vaccinated."
    },
    "details": {
        "BCG": "Bacillus Calmette-Guérin, for Tuberculosis (TB) protection.",
        "OPV": "Oral Polio Vaccine, protects against Polio.",
        "Hepatitis B": "Protects against Hepatitis B virus infection.",
        "DPT": "Diphtheria, Pertussis (Whooping Cough), and Tetanus.",
        "Rotavirus": "Protects against Rotavirus diarrhea.",
        "PCV": "Pneumococcal Conjugate Vaccine, protects against certain types of pneumonia and meningitis.",
        "MR": "Measles and Rubella vaccine.",
        "Td": "Tetanus and adult Diphtheria vaccine."
    }
}
VACCINE_KEYWORDS = ['vaccine', 'vaccination', 'immunization', 'schedule', 'dpt', 'opv', 'bcg', 'polio', 'measles', 'rubella', 'hepatitis', 'tetanus', 'rota', 'pcv', 'mr', 'td', 'टीका', 'टीकाकरण']


# --- Core Functions ---

def is_vaccine_question(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in VACCINE_KEYWORDS)

def get_vaccine_response(language_code="en"):
    # This function formats the schedule into a clear, readable string.
    # In a real application, this would also be translated.
    schedule_info = VACCINATION_SCHEDULE['schedule']
    details_info = VACCINATION_SCHEDULE['details']
    
    response = f"**{VACCINATION_SCHEDULE['title']}**\n\n"
    response += "Here is the recommended vaccination schedule for children and pregnant women:\n\n"
    for age, vaccines in schedule_info.items():
        response += f"- **At {age}:** {vaccines}\n"
    
    response += "\n**Details about Vaccines:**\n"
    for vaccine, detail in details_info.items():
        response += f"- **{vaccine}:** {detail}\n"
        
    return response

def find_best_chunks(question, index, chunks, top_k=3):
    if not index or not chunks or not GENAI_AVAILABLE: return []
    try:
        query_embedding_result = genai.embed_content(model=EMBEDDING_MODEL, content=question, task_type="RETRIEVAL_QUERY")
        query_embedding = np.array(query_embedding_result['embedding']).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

# --- UPDATED FUNCTION WITH CONVERSATIONAL CONTEXT ---
def get_health_response(question, language_code="en", history=[]): # Added history parameter
    if not GENAI_AVAILABLE:
        return "The AI health assistant is currently unavailable."

    # PRIORITY 1: Check for vaccine-related questions (applies to the current question only)
    if is_vaccine_question(question):
        # We don't need history for a direct, factual lookup like this.
        return get_vaccine_response(language_code)

    # PRIORITY 2: Use the book for general health queries
    language_name = LANGUAGE_MAP.get(language_code, "English")
    context_from_book = find_best_chunks(question, faiss_index, text_chunks)
    context_str = "\n".join(context_from_book)

    # The system prompt is now separate to define the AI's core behavior and handle follow-ups.
    system_prompt = f"""
        You are an AI Health Advisor. Your persona is supportive, knowledgeable, and cautious. Your role is to provide helpful, safe, and informative content based primarily on the provided context from a medical encyclopedia.
        Your entire response, including all sections and disclaimers, MUST be in {language_name}.
        Your tone must be concise, clear, supportive, and never rude or demotivating.

        When a user provides a vague follow-up like "ok proceed" or "tell me more", you MUST ask for clarification based on the conversational history. For example, if your previous response was a list of points, ask "Which point would you like me to elaborate on?". Do not generate a new, generic health response.

        **Context from Medical Encyclopedia for the CURRENT question:**
        ---
        {context_str if context_str else "No specific context was found in the book for this query. Answer based on your general knowledge but be extremely cautious and prioritize safety."}
        ---

        **RESPONSE STRUCTURE AND INSTRUCTIONS (for new health queries):**
        If the user is asking a new health question (not a follow-up), you MUST structure your response as follows:

        1.  **Mandatory Disclaimer (Start with this):** Begin your response with the following disclaimer, exactly as written: "**Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health.**"
        2.  **Direct Answer:** Directly answer the user's question based on the provided 'Context from Medical Encyclopedia'.
        3.  **Structured Advice Sections:** Provide these sections using Markdown:
            * **Key Precautions:** List 2-3 important precautions.
            * **Dietary Suggestions:** Briefly mention "Foods to Include" and "Foods to Limit".
            * **When to Consult a Doctor:** State specific symptoms for when to see a doctor and suggest specialist types if applicable.
        4.  **Medication Information (Conditional & Extremely Cautious):** ONLY for minor ailments, mention common over-the-counter medication, preceded by this EXACT warning: "**Medical Advisory: The following is for informational purposes ONLY, in a scenario where a doctor is not immediately accessible. Self-medication is risky and should be avoided. Always consult a healthcare professional before taking any medication.**"
        5.  **Closing:** Conclude your response by asking an engaging, supportive question, like: "Would you like to know more about any of these points?"
    """
    try:
        model = genai.GenerativeModel(
            GENERATIVE_MODEL,
            system_instruction=system_prompt
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(question) # Send only the new question
        return response.text if hasattr(response, 'text') and response.text else "I couldn't generate a response."
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return "An error occurred while trying to get a response."


# --- Flask Routes ---
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")
    language = data.get("language", "en")
    history = data.get("history", []) # Get history from request
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    # Pass history to the function
    answer = get_health_response(user_question, language, history)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

