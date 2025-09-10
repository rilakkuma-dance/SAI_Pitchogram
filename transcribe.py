import whisper
import librosa

# Load audio using librosa instead of ffmpeg
audio, sr = librosa.load("thai_khaao.mp3", sr=16000)

# Load whisper model
model = whisper.load_model("turbo")

# Transcribe
result = model.transcribe(audio, language="th", fp16=False)
print(result["text"])