import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
import os
import sys

# === CONFIG ===
AUDIO_FILE = "mandarin_test.wav"
MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

print("Loading model, feature extractor, and tokenizer...")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_NAME)

# === Load audio file ===
if not os.path.exists(AUDIO_FILE):
    print(f"Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

audio_input, sample_rate = torchaudio.load(AUDIO_FILE)
print(f"Original sample rate: {sample_rate} Hz, channels: {audio_input.shape[0]}")

# === Resample if needed ===
if sample_rate != 16000:
    print("Resampling to 16 kHz...")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)
    sample_rate = 16000

# === Convert to mono if needed ===
waveform = audio_input.mean(dim=0)
print(f"Waveform shape (mono): {waveform.shape}")

# === Prepare input tensor ===
input_values = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_values

# === Inference ===
print("Running phoneme recognition...")
with torch.no_grad():
    logits = model(input_values).logits

# === Decode phonemes ===
predicted_ids = torch.argmax(logits, dim=-1)
phonemes = tokenizer.batch_decode(predicted_ids)
print("\n=== Phoneme Transcription ===")
print(phonemes[0])
