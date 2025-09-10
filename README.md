# Health Chatbot using Flask & Gemini AI

## Overview
This is a simple **Flask-based Health Chatbot** that uses Google's **Gemini AI** to provide responses to health-related questions. The chatbot is designed to answer queries about various health topics such as diseases, symptoms, treatments, nutrition, and fitness.

## Features
- Uses **Gemini AI (Generative Model)** for generating responses.
- Recognizes health-related questions using keyword filtering.
- **Flask Web App** for easy interaction.
- Supports **JSON-based API** for chatbot integration.

## Tech Stack
- **Python** (Flask)
- **Google Gemini AI** (Generative Model API)
- **HTML/CSS/JavaScript** (for frontend)

## Project Structure
```
/health-chatbot
│── app.py                 # Main Flask app
│── templates/
│   └── index.html         # Frontend UI
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

## Getting Started

### 🔹 1. Clone the repository
```sh
git clone https://github.com/jeslipriya/AI-Health-Chatbot.git
cd AI-Health-Chatbot
```

### 🔹 2. Create a virtual environment (optional but recommended)
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 🔹 3. Install dependencies
```sh
pip install -r requirements.txt
```


### 🔹 4. Set up API Key
For local development, create a `.env` file with:
```
GEMINI_API_KEY=your_actual_google_gemini_api_key
```
For deployment on Render:
- Go to your Render dashboard
- Open your web service settings
- Add an environment variable:
	- Key: `GEMINI_API_KEY`
	- Value: your actual API key (no quotes)


### 🔹 5. Run the Flask app
```sh
python app.py
```
Access the chatbot at **http://127.0.0.1:5000/** in your browser.

---

## 🚀 Deploying to Render (Free)
1. Push your code to GitHub.
2. Go to [https://render.com](https://render.com) and create a new Web Service.
3. Connect your GitHub repo and select your project.
4. Render will auto-detect your `.render.yaml` and set up the build/start commands.
5. Add your `GEMINI_API_KEY` as an environment variable in the Render dashboard.
6. Click "Manual Deploy" to deploy your app.
7. Your app will be live at the provided Render URL.

**Note:** Free Render instances may spin down with inactivity, causing a delay on the first request.

## API Endpoint
The chatbot also supports a **POST** request for API integration:
```
POST /ask
Request Body: { "question": "What are the symptoms of flu?" }
Response: { "answer": "Flu symptoms include fever, cough, sore throat..." }
```

## Customization
- Modify `HEALTH_KEYWORDS` in `app.py` to improve keyword filtering.
- Edit `index.html` for a better UI experience.

## Contributing
Pull requests are welcome! Feel free to suggest improvements.

---
**Let's build smarter health solutions together!**



# to activate the environment
venv\Scripts\activate