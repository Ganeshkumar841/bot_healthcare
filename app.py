from flask import Flask, render_template, request, jsonify
import os
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

API_KEY = "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4"
if API_KEY == "YOUR_API_KEY_HERE":
    print("Warning: API_KEY is not set as an environment variable. Using a placeholder.")

if GENAI_AVAILABLE:
    genai.configure(api_key=API_KEY)

HEALTH_KEYWORDS = [
    # Original Keywords (Symptoms & Core Conditions)
    "fever", "cold", "flu", "diabetes", "hypertension", "cancer", "asthma",
    "allergy", "infection", "migraine", "stroke", "cough", "headache", "nausea",
    "fatigue", "dizziness", "sore throat", "rash", "inflammation", "swelling",
    "pain", "anxiety", "depression", "stress", "sleep disorders",

    # Infectious Diseases
    "pneumonia", "bronchitis", "tuberculosis", "hepatitis", "hiv", "aids",
    "malaria", "dengue", "chikungunya", "typhoid", "cholera", "meningitis",
    "sepsis", "leprosy", "tetanus", "chickenpox", "measles", "mumps", "rabies",
    "urinary tract infection", "uti",

    # Cardiovascular Diseases (Heart & Blood Vessel)
    "heart attack", "myocardial infarction", "heart failure", "arrhythmia",
    "coronary artery disease", "atherosclerosis", "aneurysm", "high cholesterol",
    "deep vein thrombosis", "pulmonary embolism",

    # Respiratory Diseases (Lungs)
    "copd", "chronic obstructive pulmonary disease", "emphysema", "cystic fibrosis",
    "sleep apnea",

    # Gastrointestinal Diseases (Digestive)
    "ulcer", "gastritis", "acid reflux", "gerd", "celiac disease", "crohn's disease",
    "ulcerative colitis", "irritable bowel syndrome", "ibs", "gallstones",
    "pancreatitis", "cirrhosis", "appendicitis", "hemorrhoids",

    # Neurological Disorders (Brain & Nerves)
    "epilepsy", "seizure", "alzheimer's", "parkinson's", "dementia",
    "multiple sclerosis", "ms", "amyotrophic lateral sclerosis", "als", "bell's palsy",

    # Endocrine & Metabolic Diseases
    "thyroid", "hyperthyroidism", "hypothyroidism", "goiter", "osteoporosis",
    "gout", "obesity", "metabolic syndrome",

    # Autoimmune Diseases
    "rheumatoid arthritis", "lupus", "psoriasis", "vitiligo",

    # Musculoskeletal & Skin Diseases
    "arthritis", "osteoarthritis", "fibromyalgia", "scoliosis",
    "carpal tunnel syndrome", "eczema", "acne", "rosacea", "melanoma", "shingles",

    # Cancers (Specific Types)
    "leukemia", "lymphoma", "lung cancer", "breast cancer", "prostate cancer",
    "colon cancer", "skin cancer", "ovarian cancer", "cervical cancer", "brain tumor",

    # Mental & Behavioral Disorders
    "bipolar disorder", "schizophrenia", "ocd", "obsessive-compulsive disorder",
    "ptsd", "post-traumatic stress disorder", "eating disorder", "anorexia", "bulimia",
    "adhd", "attention-deficit/hyperactivity disorder",

    # Kidney & Urological Diseases
    "kidney stones", "kidney disease", "kidney failure", "polycystic kidney disease", "pkd",
    "bladder infection", "incontinence",

    # Blood Disorders
    "anemia", "hemophilia", "sickle cell disease", "thalassemia",

    # Eye & Ear Conditions
    "glaucoma", "cataracts", "macular degeneration", "conjunctivitis", "pink eye",
    "tinnitus", "vertigo"
]

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

def is_health_related(question):
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in HEALTH_KEYWORDS)

def get_health_response(question, language_code="en"):
    if not GENAI_AVAILABLE:
        return "The AI health assistant is currently unavailable because the required AI module is not installed. Please contact the administrator."
    try:
        if not is_health_related(question):
            if language_code == "es":
                return "Soy un asistente centrado en la salud. Por favor, hazme preguntas relacionadas con la salud, el bienestar y la medicina."
            if language_code == "hi":
                return "मैं एक स्वास्थ्य-केंद्रित सहायक हूँ। कृपया मुझसे स्वास्थ्य, कल्याण और चिकित्सा से संबंधित प्रश्न पूछें।"
            if language_code == "te":
                return "నేను ఆరోగ్య-కేంద్రీకృత సహాయకుడిని. దయచేసి నన్ను ఆరోగ్యం, శ్రేయస్సు మరియు వైద్యానికి సంబంధించిన ప్రశ్నలు అడగండి."
            if language_code == "ta":
                return "நான் ஒரு சுகாதாரத்தை மையமாகக் கொண்ட உதவியாளர். தயவுசெய்து என்னிடம் சுகாதாரம், ஆரோக்கியம் மற்றும் மருத்துவம் தொடர்பான கேள்விகளைக் கேளுங்கள்."
            if language_code == "or":
                return "ମୁଁ ଜଣେ ସ୍ୱାସ୍ଥ୍ୟ-କେନ୍ଦ୍ରିତ ସହାୟକ | ଦୟାକରି ମୋତେ ସ୍ୱାସ୍ଥ୍ୟ, ସୁସ୍ଥତା ଏବଂ ଚିକିତ୍ସା ସମ୍ବନ୍ଧୀୟ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ |"
            return "I am a health-focused assistant. Please ask me questions related to health, wellness, and medicine."

        language_name = LANGUAGE_MAP.get(language_code, "English")

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        You are an AI Health Advisor. Your role is to provide helpful and informative content.
        You are not a doctor.
        Your entire response, including the disclaimer, MUST be in {language_name}.

        Start your response with a disclaimer. The disclaimer must say: "Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health." Make this disclaimer bold.
        
        After the disclaimer, answer the following user question: "{question}"
        """

        response = model.generate_content(prompt)

        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "I apologize, but I couldn't generate a response at this time. Please try again."

    except Exception as e:
        # Handle Gemini API quota exceeded error
        if hasattr(e, 'args') and e.args and '429' in str(e.args[0]):
            return ("You have exceeded your current quota for the Gemini API. "
                    "Please wait for your quota to reset or check your plan and billing details. "
                    "See: https://ai.google.dev/gemini-api/docs/rate-limits")
        print(f"An error occurred: {e}")
        return "An error occurred while trying to get a response. Please check the server logs."

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Invalid request. 'question' is required."}), 400
    
    user_question = data.get("question")
    language = data.get("language", "en")
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = get_health_response(user_question, language)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)