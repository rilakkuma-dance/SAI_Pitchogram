import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import os
from typing import List, Dict, Optional

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("pypinyin not found. Install with: pip install pypinyin")

class Wav2Vec2PypinyinToneClassifier:
    """Complete tone classification using wav2vec2 + pypinyin dictionary lookup"""
    
    def __init__(self, wav2vec2_model="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"):
        """Initialize wav2vec2 and pypinyin"""
        
        if not PYPINYIN_AVAILABLE:
            raise ImportError("pypinyin is required. Install with: pip install pypinyin")
        
        # Load Chinese wav2vec2
        print("Loading Chinese wav2vec2...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()
        
        self.vocab = self.processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Wav2vec2 loaded successfully on {self.device}")
        print("Pypinyin tone lookup ready")
    
    def detect_characters_with_timing(self, audio_input):
        """Detect Chinese characters with timing using wav2vec2
        
        Args:
            audio_input: str (file path), np.ndarray, or torch.Tensor
        """
        try:
            # Process different input types
            if isinstance(audio_input, str):
                # File path - load using librosa
                audio = librosa.load(audio_input, sr=16000)[0]
            elif isinstance(audio_input, torch.Tensor):
                # Convert tensor to numpy and resample to 16kHz
                audio = audio_input.cpu().numpy()
                if audio.ndim > 1:
                    audio = audio[0]  # Take first channel
                # Assume input needs resampling to 16kHz
                audio = librosa.resample(audio, orig_sr=len(audio)//10 if len(audio) > 160000 else 16000, target_sr=16000)
            elif isinstance(audio_input, np.ndarray):
                # Numpy array - resample to 16kHz
                audio = audio_input.copy()
                if audio.ndim > 1:
                    audio = audio[0]  # Take first channel
                # Assume input needs resampling to 16kHz
                audio = librosa.resample(audio, orig_sr=len(audio)//10 if len(audio) > 160000 else 16000, target_sr=16000)
            else:
                raise ValueError(f"Unsupported input type: {type(audio_input)}")
            
            audio = librosa.util.normalize(audio)
            
            # Get wav2vec2 predictions
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            print(f"Wav2vec2 transcription: '{transcription}'")
            
            # Extract character timings
            char_timings = self.extract_character_timings(predicted_ids)
            
            return transcription, char_timings
            
        except Exception as e:
            print(f"Character detection error: {e}")
            return "", []
    
    def extract_character_timings(self, predicted_ids):
        """Extract character-level timing information"""
        frame_predictions = predicted_ids.cpu().numpy()[0]
        frame_duration = 0.02  # wav2vec2 frame rate: 50Hz
        
        char_timings = []
        current_char = None
        char_start = 0
        
        for i, token_id in enumerate(frame_predictions):
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens and empty tokens
                if token in ['<pad>', '<s>', '</s>', '<unk>', '|', '[PAD]', ' ', '']:
                    continue
                
                # Only process Chinese characters (filter out artifacts)
                if token != current_char and token.strip() and self.is_chinese_character(token):
                    # End previous character
                    if current_char is not None and current_char.strip():
                        duration = (i - char_start) * frame_duration
                        if duration > 0.02:  # Minimum 20ms duration
                            char_timings.append({
                                'char': current_char,
                                'start_time': char_start * frame_duration,
                                'end_time': i * frame_duration,
                                'duration': duration
                            })
                    
                    # Start new character
                    current_char = token
                    char_start = i
        
        # Handle last character
        if current_char is not None and current_char.strip():
            duration = (len(frame_predictions) - char_start) * frame_duration
            if duration > 0.02:
                char_timings.append({
                    'char': current_char,
                    'start_time': char_start * frame_duration,
                    'end_time': len(frame_predictions) * frame_duration,
                    'duration': duration
                })
        
        return char_timings
    
    def is_chinese_character(self, char):
        """Check if character is Chinese"""
        if not char:
            return False
        # Chinese character Unicode ranges
        for c in char:
            if not ('\u4e00' <= c <= '\u9fff' or  # CJK Unified Ideographs
                   '\u3400' <= c <= '\u4dbf' or   # CJK Extension A
                   '\uf900' <= c <= '\ufaff'):     # CJK Compatibility Ideographs
                return False
        return True
    
    def get_tone_from_pypinyin(self, char):
        """Get tone number from Chinese character using pypinyin"""
        try:
            # Get pinyin with tone numbers
            result = pinyin(char, style=Style.TONE3, strict=False)
            
            if result and result[0]:
                pinyin_with_tone = result[0][0]
                
                # Extract tone number from end of pinyin
                if pinyin_with_tone and pinyin_with_tone[-1].isdigit():
                    tone = int(pinyin_with_tone[-1])
                    # Handle tone 5 (neutral tone)
                    if tone == 5:
                        tone = 0  # Convert to 0 for neutral tone
                    return tone, pinyin_with_tone[:-1]  # Return tone and pinyin without number
                else:
                    # No tone number found, might be neutral tone
                    return 0, pinyin_with_tone  # 0 for neutral tone
            
            return None, None
            
        except Exception as e:
            print(f"Pypinyin error for '{char}': {e}")
            return None, None
    
    def get_multiple_pronunciations(self, char):
        """Get all possible pronunciations for a character"""
        try:
            # Get all possible pronunciations
            results = pinyin(char, style=Style.TONE3, heteronym=True, strict=False)
            
            pronunciations = []
            if results and results[0]:
                for pronunciation in results[0]:
                    if pronunciation and pronunciation[-1].isdigit():
                        tone = int(pronunciation[-1])
                        if tone == 5:
                            tone = 0
                        pinyin_base = pronunciation[:-1]
                        pronunciations.append({
                            'pinyin': pinyin_base,
                            'tone': tone,
                            'full_pinyin': pronunciation
                        })
                    else:
                        pronunciations.append({
                            'pinyin': pronunciation,
                            'tone': 0,  # Neutral tone
                            'full_pinyin': pronunciation
                        })
            
            return pronunciations
            
        except Exception as e:
            print(f"Multiple pronunciation error for '{char}': {e}")
            return []
    
    def classify_tones_from_characters(self, char_timings, show_alternatives=False):
        """Classify tones using pypinyin for detected characters"""
        results = []
        
        for i, char_timing in enumerate(char_timings):
            char = char_timing['char']
            
            # Get tone from pypinyin
            tone, pinyin_base = self.get_tone_from_pypinyin(char)
            
            if tone is not None:
                result = {
                    'syllable_idx': i,
                    'character': char,
                    'pinyin': pinyin_base,
                    'predicted_tone': tone,
                    'confidence': 1.0,  # Dictionary lookup is definitive
                    'start_time': char_timing['start_time'],
                    'end_time': char_timing['end_time'],
                    'duration': char_timing['duration'],
                    'method': 'pypinyin_lookup'
                }
                
                # Add alternative pronunciations if requested
                if show_alternatives:
                    alternatives = self.get_multiple_pronunciations(char)
                    if len(alternatives) > 1:
                        result['alternatives'] = alternatives
                
                results.append(result)
            else:
                # Unknown character
                result = {
                    'syllable_idx': i,
                    'character': char,
                    'pinyin': 'unknown',
                    'predicted_tone': None,
                    'confidence': 0.0,
                    'start_time': char_timing['start_time'],
                    'end_time': char_timing['end_time'],
                    'duration': char_timing['duration'],
                    'method': 'unknown'
                }
                results.append(result)
        
        return results
    
    # chris: I know this is tones but just making the interface consistent!
    def predict_tone(self, audio_input, show_alternatives=False):
        """Main function: detect characters and classify tones
        
        Args:
            audio_input: str (file path), np.ndarray, or torch.Tensor
            show_alternatives: bool
        """
        
        if isinstance(audio_input, str):
            print(f"Processing: {os.path.basename(audio_input)}")
        else:
            print(f"Processing: {type(audio_input).__name__} array")
        
        # Step 1: Detect characters with timing
        transcription, char_timings = self.detect_characters_with_timing(audio_input)
        
        if not char_timings:
            print("No Chinese characters detected")
            return None
        
        print(f"Detected characters: {[ct['char'] for ct in char_timings]}")
        
        # Step 2: Classify tones using direct character-to-tone mapping
        results = self.classify_tones_from_characters(char_timings, show_alternatives)
        
        return results
    
    def format_results(self, results, show_details=True):
        """Format results for display"""
        if not results:
            return "No results available"
        
        output = []
        output.append("Wav2vec2 + Pypinyin Tone Classification:")
        output.append("=" * 55)
        output.append(f"Total syllables: {len(results)}")
        output.append("")
        
        successful_results = [r for r in results if r['predicted_tone'] is not None]
        
        for result in results:
            char = result['character']
            pinyin = result.get('pinyin', 'unknown')
            tone = result['predicted_tone']
            method = result['method']
            
            if tone is not None:
                output.append(f"Character: {char}")
                output.append(f"  Pinyin: {pinyin}")
                output.append(f"  Tone: T{tone}" if tone > 0 else "  Tone: Neutral")
                output.append(f"  Confidence: {result['confidence']:.3f}")
                output.append(f"  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
                if show_details:
                    output.append(f"  Method: {method}")
                
                # Show alternatives if available
                if 'alternatives' in result and len(result['alternatives']) > 1:
                    output.append(f"  Alternative pronunciations:")
                    for alt in result['alternatives']:
                        alt_tone = alt['tone']
                        tone_str = f"T{alt_tone}" if alt_tone > 0 else "Neutral"
                        output.append(f"    {alt['full_pinyin']} ({tone_str})")
                
            else:
                output.append(f"Character: {char}")
                output.append(f"  Tone: UNKNOWN")
                output.append(f"  Method: {method}")
            
            output.append("")
        
        # Summary
        if successful_results:
            chars = [r['character'] for r in successful_results]
            pinyins = [r.get('pinyin', 'unknown') for r in successful_results]
            tones = []
            
            for r in successful_results:
                tone = r['predicted_tone']
                if tone is not None and tone > 0:
                    tones.append(str(tone))
                else:
                    tones.append('0')  # Neutral tone
            
            output.append("SUMMARY:")
            output.append(f"Characters: {''.join(chars)}")
            output.append(f"Pinyin: {' '.join(pinyins)}")
            output.append(f"Tones: {' '.join(tones)}")
            
            # Create pinyin with tone numbers
            pinyin_with_tones = []
            for pinyin_base, tone in zip(pinyins, tones):
                if tone == '0':
                    pinyin_with_tones.append(f"{pinyin_base}5")  # Neutral tone as 5
                else:
                    pinyin_with_tones.append(f"{pinyin_base}{tone}")
            
            output.append(f"Full pinyin: {' '.join(pinyin_with_tones)}")
            output.append(f"Success rate: {len(successful_results)}/{len(results)} characters")
        
        return "\n".join(output)

def main():
    """Test the system with audio file auto-detection"""
    
    if not PYPINYIN_AVAILABLE:
        print("Please install pypinyin first:")
        print("pip install pypinyin")
        return
    
    print("=" * 60)
    print("WAV2VEC2 + PYPINYIN TONE CLASSIFICATION SYSTEM")
    print("=" * 60)
    
    # Initialize classifier
    try:
        classifier = Wav2Vec2PypinyinToneClassifier()
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        return
    
    # Process audio file
    audio_path = input("Enter audio file path: ").strip()
    
    if not os.path.exists(audio_path):
        print("Audio file not found")
        return
    
    results = classifier.predict_tone(audio_path, show_alternatives=True)
    
    if results:
        print("\n" + classifier.format_results(results))
    else:
        print("No results obtained")

if __name__ == "__main__":
    main()