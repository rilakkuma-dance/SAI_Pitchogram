from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer

import torchaudio
import torch
import panphon.distance
import difflib
import numpy as np

class PhonemeHandler:
    def __init__(self, model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.feature_encoder = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
            self.sample_rate = 16000
        except Exception as e:
            raise e
        
    def _transcribe_audio(self, waveform: torch.Tensor) -> str:
        # Feature encoder
        inputs = self.feature_encoder(
            waveform, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Raw transcription: '{transcription}'")
                
        # Clean up the transcription
        cleaned_transcription = transcription.strip()
        return cleaned_transcription

    def transcribe_audio_from_file(self, file_path: str) -> str:
        try:
            audio_input, sample_rate = torchaudio.load(file_path)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                audio_input = resampler(audio_input)
                sample_rate = self.sample_rate

            # Make the audio mono as well
            waveform = audio_input.mean(dim=0)

            return self._transcribe_audio(waveform)
        except Exception as e:
            raise e
    
    def transcribe_audio_from_numpy(self, audio_array: np.ndarray) -> str:
        try:
            # Ensure the audio array is mono
            if audio_array.ndim > 1 and audio_array.shape[0] > 1:
                waveform = torch.tensor(audio_array.mean(axis=0))
            else:
                waveform = torch.tensor(audio_array)

            return self._transcribe_audio(waveform)
        except Exception as e:
            raise e

    def transcribe_audio_from_tensor(self, audio_tensor: torch.Tensor) -> str:
        try:
            # Ensure the audio tensor is mono
            if audio_tensor.dim() > 1 and audio_tensor.size(0) > 1:
                waveform = audio_tensor.mean(dim=0)
            else:
                waveform = audio_tensor

            return self._transcribe_audio(waveform)
        except Exception as e:
            raise e
    
    def _process_transcription(self, transcription: str) -> list[str]:
        # Split the transcription into phonemes based on spaces
        phonemes = transcription.strip().split()
        return phonemes

class PhonemeAnalyzer:
    """Analyzes phoneme sequences and provides detailed feedback"""

    def __init__(self, reference_phonemes: str):
        # Because we're switching to Wav2Vec2CTCTokenizer, the phonemes will now
        # come in a single string. We can split them normally.
        self.reference_phonemes = self.normalize_phoneme(reference_phonemes)

    def normalize_phoneme(self, phonemes: str) -> str:
        """Normalize eSpeak phonemes by removing tone markers"""
        # Remove tone markers and diacritics for core comparison
        core = phonemes.lower().strip()
        # Remove common tone markers
        tone_markers = set(['˥', '˦', '˧', '˨', '˩', '¹', '²', '³', '⁴', '⁵', '1', '2', '3', '4', '5', ':'])
        core = ''.join(filter(lambda c: c not in tone_markers, core))
        return core

    def phoneme_similarity(self, test: str, reference: str | None = None) -> float:
        """Calculate similarity between two phonemes (0.0 to 1.0)"""
        if reference is None:
            reference = self.reference_phonemes
        
        p1, p2 = self.normalize_phoneme(test), self.normalize_phoneme(reference)

        if p1 == p2:
            return 1.0
            
        if not p1 or not p2:
            return 0.0

        dist_measurement = panphon.distance.Distance()
        max_len = max(len(p1), len(p2))

        edit_distance = dist_measurement.weighted_feature_edit_distance_div_maxlen([p1], [p2])
        return (1 - (edit_distance / max_len))
    
    def align_phonemes(self, detected: str, target: str | None = None):
        """Align detected phonemes with target using dynamic programming"""
        if target is None:
            target = self.reference_phonemes

        if not detected or not target:
            return [], []
        
        # Convert strings to character lists for phoneme-level analysis
        det_chars = list(detected)
        tgt_chars = list(target)
        
        # Use difflib for initial alignment
        matcher = difflib.SequenceMatcher(None, det_chars, tgt_chars)
        aligned_detected = []
        aligned_target = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Perfect matches
                for k in range(i2 - i1):
                    aligned_detected.append(det_chars[i1 + k])
                    aligned_target.append(tgt_chars[j1 + k])
            elif tag == 'replace':
                # Substitutions
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    det_char = det_chars[i1 + k] if i1 + k < i2 else None
                    tgt_char = tgt_chars[j1 + k] if j1 + k < j2 else None
                    aligned_detected.append(det_char)
                    aligned_target.append(tgt_char)
            elif tag == 'delete':
                # Deletions (detected has extra)
                for k in range(i2 - i1):
                    aligned_detected.append(det_chars[i1 + k])
                    aligned_target.append(None)
            elif tag == 'insert':
                # Insertions (target has extra)
                for k in range(j2 - j1):
                    aligned_detected.append(None)
                    aligned_target.append(tgt_chars[j1 + k])
        
        return aligned_detected, aligned_target

    def analyze_pronunciation(self, detected: str, reference: str | None = None):
        """Analyze pronunciation and return detailed feedback"""
        if not reference:
            reference = self.reference_phonemes

        if not detected and not reference:
            return [], 0.0
        
        if not detected:
            # No detection - all target phonemes are missing
            return [{'target': c, 'detected': None, 'similarity': 0.0, 'status': 'missing'} 
                   for c in reference], 0.0
        
        if not reference:
            # No target - all detected phonemes are extra
            return [{'target': None, 'detected': c, 'similarity': 0.0, 'status': 'extra'} 
                   for c in detected], 0.0
        
        detected = self.normalize_phoneme(detected)
        reference = self.normalize_phoneme(reference)

        print("Analyzing pronunciation: ", detected, reference)
        
        aligned_detected, aligned_target = self.align_phonemes(detected, reference)

        print(aligned_detected, aligned_target)
        
        results = []
        total_similarity = 0.0
        valid_comparisons = 0
        
        for det, tgt in zip(aligned_detected, aligned_target):
            if det is None and tgt is not None:
                # Missing phoneme
                valid_comparisons += 1
                results.append({
                    'target': tgt,
                    'detected': None,
                    'similarity': 0.0,
                    'status': 'missing'
                })
            elif det is not None and tgt is None:
                # Extra phoneme
                results.append({
                    'target': None,
                    'detected': det,
                    'similarity': 0.0,
                    'status': 'extra'
                })
            else:
                # Both present - calculate similarity
                similarity = self.phoneme_similarity(det, tgt)
                total_similarity += similarity
                valid_comparisons += 1
                
                if similarity >= 0.8:
                    status = 'correct'
                elif similarity >= 0.5:
                    status = 'close'
                else:
                    status = 'incorrect'
                
                results.append({
                    'target': tgt,
                    'detected': det,
                    'similarity': similarity,
                    'status': status
                })

        print(total_similarity, valid_comparisons)
        
        # Calculate overall score
        overall_score = self.phoneme_similarity(detected, reference)
        print("Overall_score:", overall_score)

        return results, max(0, overall_score)