from flask import Flask, render_template, request, jsonify
import google.generativeai as genai


API_KEY = "AIzaSyBK8mGnBUsRoQPVmZaITMG0KkfYkEmCUP4"  
genai.configure(api_key=API_KEY)

HEALTH_KEYWORDS = [
    "fever", "cold", "flu", 'diabetes', 'hypertension', 'cancer', 'asthma', "hi",
    'allergy', 'infection', 'migraine', 'stroke Cardiology', 'dermatology', "hello",
    'neurology', 'orthopedics', 'ophthalmology', 'psychiatry', 'endocrinology Cough',
    'headache', 'nausea', 'fatigue', 'dizziness', 'sore throat', 'rash', 'inflammation',
    'swelling', 'pain', 'antibiotics', 'painkillers', 'therapy', 'surgery', 'vaccination',
    'home remedies', 'supplements', 'Stress', 'anxiety', 'depression', 'mindfulness',
    'therapy', 'meditation', 'sleep disorders', 'vitamins', 'proteins', 'carbohydrates',
    'fiber', 'hydration', 'intermittent fasting', 'weight loss', 'diet plan', 'cardio',
    'strength', 'training', 'yoga', 'stretching', 'endurance', 'mobility', "muscle recovery"
]

def is_health_related(question):
    """Check if the question is health-related by looking for keywords."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in HEALTH_KEYWORDS)

def get_health_response(question):
    """Get health-related responses from Gemini AI."""
    try:
        if not is_health_related(question):
            return "I can only answer health-related questions. Please ask about health."

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(question)
        
        return response.text if response.text else "Sorry, I couldn't generate a response."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = get_health_response(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)