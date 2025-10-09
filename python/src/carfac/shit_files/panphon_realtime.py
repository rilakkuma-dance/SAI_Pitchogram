import panphon.distance
import speech_recognition as sr
from phonemizer import phonemize

# PanPhon setup
dist = panphon.distance.Distance()
max_dist = 24 * max(4, 4)

# SpeechRecognition setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Target phoneme sequence (IPA example)
target_phonemes = ["x", "ie", "x", "ie"]

print("🎤 Speak now... (Ctrl+C to stop)")

while True:
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        
        # Speech-to-text
        spoken_text = recognizer.recognize_google(audio)
        print(f"You said: {spoken_text}")

        # Convert recognized text to phonemes (IPA)
        spoken_phonemes = phonemize(spoken_text, language="en-us").split()
        print(f"Phonemes: {spoken_phonemes}")

        # Compare with target
        example = dist.phoneme_error_rate(target_phonemes, spoken_phonemes)
        similarity = (1 - example / max_dist) * 100
        print(f"Target: {target_phonemes}")
        print(f"Distance: {example}, Similarity: {similarity:.2f}%\n")

    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print(f"Error: {e}\n")
