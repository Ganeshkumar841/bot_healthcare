from flask import Flask, render_template, request, jsonify
import os
import faiss
import numpy as np

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- Configuration ---
API_KEY = "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4"
if GENAI_AVAILABLE and API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=API_KEY)
else:
    GENAI_AVAILABLE = False
    print("Warning: API_KEY is not set or google.generativeai is not installed.")

# Models
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash"

# File paths for the processed book data
FAISS_INDEX_PATH = "health_book.index"
TEXT_CHUNKS_PATH = "health_book_chunks.txt"

# --- Load RAG Data (Book Index and Chunks) ---
faiss_index = None
text_chunks = []

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TEXT_CHUNKS_PATH, "r", encoding="utf-8") as f:
            text_chunks = f.read().split("\n---\n")
        print("Successfully loaded FAISS index and text chunks.")
    except Exception as e:
        print(f"Error loading RAG files: {e}")
        faiss_index = None # Disable RAG if loading fails
else:
    print("Warning: FAISS index or text chunks file not found. The bot will run without book context.")

# --- Health Keywords and Language ---
HEALTH_KEYWORDS = [ "fever", "cold", "flu", "diabetes", "hypertension", "cancer", "asthma", "allergy", "infection", "migraine", "stroke", "cough", "headache", "nausea", "fatigue", "dizziness", "sore throat", "rash", "inflammation", "swelling", "pain", "anxiety", "depression", "stress", "sleep disorders" ] # Truncated for brevity
LANGUAGE_MAP = { "en": "English", "es": "Spanish", "hi": "Hindi", "fr": "French", "de": "German", "te": "Telugu", "ta": "Tamil", "or": "Odia" }

# --- Core Functions ---

def is_health_related(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in HEALTH_KEYWORDS)

def find_best_chunks(question, index, chunks, top_k=3):
    """Finds the most relevant text chunks from the book for a given question."""
    if not index or not chunks:
        return []
    
    try:
        # Generate embedding for the user's question (query)
        query_embedding_result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array(query_embedding_result['embedding']).astype('float32').reshape(1, -1)
        
        # Search the FAISS index
        distances, indices = index.search(query_embedding, top_k)
        
        # Retrieve the corresponding text chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

def get_health_response(question, language_code="en"):
    if not GENAI_AVAILABLE:
        return "The AI health assistant is currently unavailable."
    
    language_name = LANGUAGE_MAP.get(language_code, "English")

    # Step 1: Find relevant context from the book
    context_from_book = find_best_chunks(question, faiss_index, text_chunks)
    context_str = "\n".join(context_from_book)

    # Step 2: Create a detailed prompt for the AI
    prompt = f"""
        You are an AI Health Advisor. Your role is to provide helpful and informative content based *primarily on the provided context from a medical encyclopedia*.
        You are not a doctor. Your entire response, including the disclaimer, MUST be in {language_name}.

        **Context from Medical Encyclopedia:**
        ---
        {context_str if context_str else "No specific context was found in the book for this query. Answer based on your general knowledge."}
        ---

        **User's Question:** "{question}"

        **Instructions:**
        1.  Start your response with a disclaimer in {language_name}. The disclaimer must say: "Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health." Make this disclaimer bold.
        2.  After the disclaimer, answer the user's question.
        3.  **Crucially, base your answer on the 'Context from Medical Encyclopedia' provided above.** If the context is relevant, synthesize the information to answer the question thoroughly.
        4.  If the context is not relevant or empty, you may use your general knowledge but state that the information was not found in the provided reference material.
    """

    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') and response.text else "I couldn't generate a response."
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return "An error occurred while trying to get a response. Please check the server logs."

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

