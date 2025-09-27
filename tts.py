# tts.py (Correct)

import asyncio
import base64
import edge_tts

# --- MODIFIED: Default voice is now a fallback ---
DEFAULT_VOICE = "en-US-AriaNeural"

async def generate_speech_base64(text: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Generates speech from text using edge-tts and returns it as a Base64 encoded string.
    
    Args:
        text (str): The text to be converted to speech.
        voice (str): The voice to use for the speech generation (e.g., "hi-IN-SwaraNeural").
        
    Returns:
        str: A Base64 encoded string representing the MP3 audio data.
    """
    if not text.strip():
        return ""

    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"An error occurred during TTS generation with voice {voice}: {e}")
        # Re-raise the exception to be handled by the main app
        raise

# Example of how to run this file standalone for testing:
if __name__ == '__main__':
    async def main():
        text_to_speak = "Hello, this is a test."
        base64_audio = await generate_speech_base64(text_to_speak)
        if base64_audio:
            with open("test_output_en.mp3", "wb") as f:
                f.write(base64.b64decode(base64_audio))
            print("Test audio (EN) successfully saved to test_output_en.mp3")

        text_to_speak_hi = "नमस्ते, यह एक परीक्षण है।"
        base64_audio_hi = await generate_speech_base64(text_to_speak_hi, voice="hi-IN-SwaraNeural")
        if base64_audio_hi:
            with open("test_output_hi.mp3", "wb") as f:
                f.write(base64.b64decode(base64_audio_hi))
            print("Test audio (HI) successfully saved to test_output_hi.mp3")

    asyncio.run(main())
    async def main():
        text_to_speak = "Hello, this is a test."
        base64_audio = await generate_speech_base64(text_to_speak)
        if base64_audio:
            with open("test_output_en.mp3", "wb") as f:
                f.write(base64.b64decode(base64_audio))
            print("Test audio (EN) successfully saved to test_output_en.mp3")

        text_to_speak_hi = "नमस्ते, यह एक परीक्षण है।"
        base64_audio_hi = await generate_speech_base64(text_to_speak_hi, voice="hi-IN-SwaraNeural")
        if base64_audio_hi:
            with open("test_output_hi.mp3", "wb") as f:
                f.write(base64.b64decode(base64_audio_hi))
            print("Test audio (HI) successfully saved to test_output_hi.mp3")

    asyncio.run(main())


