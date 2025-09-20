# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import os
import wave
import json
import subprocess

# --- NEW: vosk library for offline speech recognition ---
from vosk import Model, KaldiRecognizer

# ==============================================================================
# 2. Vosk Model Configuration
# ==============================================================================

MODEL_PATH = "vosk-model-small-en-in-0.4" 

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError
    model = Model(MODEL_PATH)
    print("Vosk model loaded successfully.")
except FileNotFoundError:
    print("="*80)
    print(f"VOSK MODEL NOT FOUND at path: '{MODEL_PATH}'")
    print("Please download a model from https://alphacephei.com/vosk/models")
    print("Unzip it, and place the folder in the root directory of this project.")
    print("="*80)
    model = None
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    model = None

# ==============================================================================
# 3. Core Transcription Function (NEW In-Memory Method)
# ==============================================================================

def transcribe_audio(file_path):
    """
    Transcribes audio by creating an in-memory pipeline from ffmpeg to Vosk,
    avoiding intermediate files to prevent locking issues on Windows.
    """
    if not model:
        raise RuntimeError("Vosk model is not loaded. Cannot perform transcription.")

    # --- Step 1: Define the FFmpeg command to output to stdout ---
    command = [
        'ffmpeg',
        '-i', file_path,       # Input file
        '-acodec', 'pcm_s16le', # Audio codec for WAV
        '-ac', '1',             # Mono channel
        '-ar', '16000',         # Sample rate required by Vosk
        '-f', 'wav',            # Output format
        '-'                     # Pipe output to stdout
    ]

    try:
        # --- Step 2: Run FFmpeg and pipe its output ---
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # --- Step 3: Feed the audio data directly into Vosk ---
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)

        # Read audio data from ffmpeg's stdout in chunks
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                # This part can be used for partial results, but we'll use the final one
                pass

        # --- Step 4: Check for errors and get final result ---
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
        if process.wait() != 0:
             print(f"FFmpeg Error: {stderr_output}")
             raise RuntimeError("FFmpeg process failed during audio conversion.")

        result = json.loads(rec.FinalResult())
        return result.get("text", "")

    except Exception as e:
        print(f"An error occurred during in-memory transcription: {e}")
        # Ensure the process is terminated if it's still running
        if 'process' in locals() and process.poll() is None:
            process.kill()
        raise

