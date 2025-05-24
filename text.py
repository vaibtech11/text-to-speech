from gtts import gTTS
from langdetect import detect
from playsound import playsound
import os

# Predefined text
predefined_text = "मेरो तो गिरधर गोपाल दूसरा न कोई जाके सिर मोर मुकुट मेरो पति सोई"

# Auto-detect the language
language = detect(predefined_text)
print(f"Detected language: {language}")

# Convert text to speech
tts = gTTS(text=predefined_text, lang=language)

# Save audio file
filename = "output_audio.mp3"
tts.save(filename)
print(f"Audio saved as {filename}")

# Play the audio file
playsound(filename)
