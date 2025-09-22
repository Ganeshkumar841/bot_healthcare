# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 2. OpenAI Whisper Configuration
# ==============================================================================

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or "YOUR_OPENAI_API_KEY" in API_KEY:
    print("="*80)
    print("OPENAI API KEY NOT FOUND or is a placeholder.")
    print("Please set a valid OPENAI_API_KEY in your .env file to use voice input.")
    print("="*80)
    client = None
else:
    try:
        client = OpenAI(api_key=API_KEY)
        print("OpenAI client initialized successfully for Whisper STT.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None

# ==============================================================================
# 3. Core Transcription Function (Using OpenAI Whisper)
# ==============================================================================

def transcribe_audio(file_path):
    """
    Transcribes audio using the OpenAI Whisper API.

    Args:
        file_path (str): The path to the audio file to transcribe.

    Returns:
        str: The transcribed text.
    """
    if not client:
        raise RuntimeError("OpenAI client is not initialized. Cannot perform transcription.")

    try:
        with open(file_path, "rb") as audio_file:
            # --- Call the Whisper API ---
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print("Successfully transcribed audio with Whisper.")
        return transcription.text

    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        # Re-raise the exception to be handled by the main app, so the user sees an error
        raise
