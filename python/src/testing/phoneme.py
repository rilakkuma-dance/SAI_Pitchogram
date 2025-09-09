# Load model directly
from transformers import AutoProcessor, AutoModelForCTC

from transformers.models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer

import soundfile as sf
import torchaudio
import torch

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
tokenizer: Wav2Vec2PhonemeCTCTokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# Load your audio file
audio_input, sample_rate = torchaudio.load("python/src/carfac/reference/vietnamese_bo.mp3")

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)
    sample_rate = 16000

# Make the audio mono as well
waveform = audio_input.mean(dim=0)

# inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# tokenize
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

# retrieve logits
with torch.no_grad():
    logits = model(inputs).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)