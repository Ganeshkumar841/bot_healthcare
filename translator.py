# This file handles language detection and translation.

# Installation required:
# pip install langdetect
# pip install deep-translator

from langdetect import detect
# MODIFICATION: Replaced googletrans with deep-translator
from deep_translator import GoogleTranslator

def detect_and_translate(text, target_lang='en'):
    """
    Detects the language of the given text and translates it to the target language if different.
    
    Args:
        text (str): The input text to process.
        target_lang (str): The target language code (e.g., 'en', 'es', 'hi').
        
    Returns:
        str: The translated (or original) text.
    """
    if not text.strip():
        return ""
        
    try:
        detected_lang = detect(text)
        
        if detected_lang != target_lang:
            # MODIFICATION: Updated translation logic for deep-translator
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated
        else:
            # Language is already the target language, no need to translate
            return text
            
    except Exception as e:
        print(f"An error occurred during language detection/translation: {e}")
        # Return the original text if detection or translation fails
        return text

