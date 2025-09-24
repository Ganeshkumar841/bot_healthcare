# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import os
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import io

# ==============================================================================
# 2. Speech Recognition Configuration
# ==============================================================================

def transcribe_audio(file_path, lang_code='en'):
    """
    Transcribes audio using Google Speech Recognition API (free tier).
    Handles audio format conversion internally from webm to WAV.

    Args:
        file_path (str): The path to the audio file to transcribe.
        lang_code (str): The language code for recognition (e.g., 'en-US', 'hi-IN').

    Returns:
        str: The transcribed text.
    """
    
    # Language code mapping for Google Speech Recognition
    lang_map = {
        'en': 'en-US',
        'hi': 'hi-IN', 
        'te': 'te-IN',
        'ta': 'ta-IN',
        'es': 'es-ES',
        'fr': 'fr-FR',
        'de': 'de-DE',
        'or': 'or-IN'
    }
    
    google_lang = lang_map.get(lang_code, 'en-US')
    
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            return ""
            
        file_size = os.path.getsize(file_path)
        print(f"Processing audio file: {file_path}, Size: {file_size} bytes")
        
        if file_size < 1000:  # Very small file
            print("Audio file too small, likely empty recording")
            return ""
        
        # Initialize recognizer with improved settings
        r = sr.Recognizer()
        r.energy_threshold = 300  # Minimum audio energy to consider for recording
        r.dynamic_energy_threshold = True
        r.dynamic_energy_adjustment_damping = 0.15
        r.dynamic_energy_ratio = 1.5
        r.pause_threshold = 0.8  # Seconds of non-speaking audio before phrase is complete
        r.operation_timeout = None  # No timeout for operations
        
        # Convert webm to wav format with better settings
        print("Converting audio format...")
        audio = AudioSegment.from_file(file_path)
        
        # Improve audio quality
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        audio = audio.set_sample_width(2)  # 16-bit samples
        
        # Normalize audio volume
        audio = audio.normalize()
        
        # Remove silence from beginning and end
        audio = audio.strip_silence(silence_len=100, silence_thresh=-40)
        
        print(f"Audio duration: {len(audio)}ms")
        
        if len(audio) < 500:  # Less than 0.5 seconds
            print("Audio too short after processing")
            return ""
        
        # Export to wav format in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Create temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_io.read())
            temp_wav_path = temp_wav.name
        
        try:
            # Load audio file
            with sr.AudioFile(temp_wav_path) as source:
                print(f"Loading audio file for recognition with language: {google_lang}")
                # Adjust for ambient noise with longer duration
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Record the entire audio
                audio_data = r.record(source)
            
            print("Attempting speech recognition...")
            
            # Try multiple recognition attempts with different approaches
            recognition_attempts = [
                # Attempt 1: Standard recognition
                lambda: r.recognize_google(audio_data, language=google_lang),
                # Attempt 2: With show_all=True to get alternatives
                lambda: r.recognize_google(audio_data, language=google_lang, show_all=False),
                # Attempt 3: Try with English if not English already
                lambda: r.recognize_google(audio_data, language='en-US') if google_lang != 'en-US' else None
            ]
            
            for i, attempt in enumerate(recognition_attempts, 1):
                try:
                    if attempt is None:
                        continue
                    print(f"Recognition attempt {i}...")
                    result = attempt()
                    if result:
                        print(f"Successfully transcribed audio using Google Speech Recognition (attempt {i})")
                        return result
                except sr.UnknownValueError:
                    print(f"Attempt {i}: Could not understand audio")
                    continue
                except sr.RequestError as e:
                    print(f"Attempt {i}: Request error: {e}")
                    continue
            
            print("All recognition attempts failed")
            return ""
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
                
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        # Return empty string instead of raising to prevent app crash
        return ""

# ==============================================================================
# 3. Alternative: Offline transcription using Whisper (if needed)
# ==============================================================================

def transcribe_audio_whisper(file_path, lang_code='en'):
    """
    Alternative transcription using OpenAI Whisper for offline processing.
    Requires: pip install openai-whisper
    
    Args:
        file_path (str): The path to the audio file to transcribe.
        lang_code (str): The language code for recognition.

    Returns:
        str: The transcribed text.
    """
    try:
        import whisper
        
        # Load model (download happens once)
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(file_path, language=lang_code if lang_code != 'en' else None)
        
        print(f"Successfully transcribed audio using Whisper with '{lang_code}' language.")
        return result["text"]
        
    except ImportError:
        raise ImportError("Whisper not installed. Run: pip install openai-whisper")
    except Exception as e:
        print(f"An error occurred during Whisper transcription: {e}")
        raise