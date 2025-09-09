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
API_KEY = os.getenv("API_KEY", "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4")
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

# THIS IS THE MISSING PIECE THAT HAS BEEN ADDED BACK
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

def get_health_response(question, language_code="en"):
    if not GENAI_AVAILABLE: return "The AI health assistant is currently unavailable."

    # PRIORITY 1: Check for vaccine-related questions
    if is_vaccine_question(question):
        return get_vaccine_response(language_code)

    # PRIORITY 2: Use the book for general health queries
    language_name = LANGUAGE_MAP.get(language_code, "English")
    context_from_book = find_best_chunks(question, faiss_index, text_chunks)
    context_str = "\n".join(context_from_book)

    prompt = f"""
        You are an AI Health Advisor. Your role is to provide helpful, safe, and informative content based primarily on the provided context from a medical encyclopedia.
        You are not a doctor. Your entire response, including the disclaimer, MUST be in {language_name}.

        **Context from Medical Encyclopedia:**
        ---
        {context_str if context_str else "No specific context was found in the book for this query. Answer based on your general knowledge but be cautious."}
        ---

        **User's Question:** "{question}"

        **Instructions:**
        1.  Start your response with a disclaimer in {language_name}. The disclaimer must say: "**Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health.**"
        2.  After the disclaimer, answer the user's question using the 'Context from Medical Encyclopedia'.
        3.  If the context is not relevant, use your general knowledge but state that specific information was not found in the reference material and strongly recommend consulting a healthcare professional.
    """
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = model.generate_content(prompt)
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
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = get_health_response(user_question, language)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

