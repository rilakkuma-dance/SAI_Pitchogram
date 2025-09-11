from transformers import AutoProcessor, AutoModelForCTC

from transformers.models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer

import torchaudio
import torch

class PhonemeHandler:
    def __init__(self, model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name)
            self.tokenizer: Wav2Vec2PhonemeCTCTokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
            self.sample_rate = 16000
        except Exception as e:
            raise e
        
    def transcribe_audio_from_file(self, file_path: str) -> list[str]:
        try:
            audio_input, sample_rate = torchaudio.load(file_path)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                audio_input = resampler(audio_input)
                sample_rate = self.sample_rate

            # Make the audio mono as well
            waveform = audio_input.mean(dim=0)

            # Tokenize
            inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

            # Retrieve logits
            with torch.no_grad():
                logits = self.model(inputs).logits

            # Take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription: list[str] = self.processor.batch_decode(predicted_ids)

            return self._process_transcription(transcription[0]) if transcription else list()
        except Exception as e:
            raise e

    def transcribe_audio_from_tensor(self, audio_tensor: torch.Tensor) -> list[str]:
        try:
            # Ensure the audio tensor is mono
            if audio_tensor.dim() > 1 and audio_tensor.size(0) > 1:
                waveform = audio_tensor.mean(dim=0)
            else:
                waveform = audio_tensor

            # Tokenize
            inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").input_values

            # Retrieve logits
            with torch.no_grad():
                logits = self.model(inputs).logits

            # Take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)

            return self._process_transcription(transcription[0]) if transcription else list()
        except Exception as e:
            raise e
    
    def _process_transcription(self, transcription: str) -> list[str]:
        # Split the transcription into phonemes based on spaces
        phonemes = transcription.strip().split()
        return phonemes