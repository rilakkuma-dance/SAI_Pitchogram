import whisper
import librosa

# Load audio using librosa instead of ffmpeg
audio, sr = librosa.load("recording_20250911_152519.wav", sr=16000)

# Load whisper model
model = whisper.load_model("large")

# Transcribe
result = model.transcribe(audio, language="zh", fp16=False)
print(result["text"])