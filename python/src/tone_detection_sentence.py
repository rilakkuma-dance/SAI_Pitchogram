import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import os
from typing import List, Dict, Optional
import re

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
    
    def detect_characters_with_timing(self, audio_path):
        """Detect Chinese characters with timing using wav2vec2"""
        try:
            # Load audio for wav2vec2
            audio = librosa.load(audio_path, sr=16000)[0]
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
    
    def segment_with_expected_syllables(self, char_timings, expected_syllables):
        """Create syllable boundaries using expected syllables and character timing"""
        if not char_timings or not expected_syllables:
            return []
        
        boundaries = []
        
        if len(char_timings) == len(expected_syllables):
            # Perfect match - use character timings directly
            for char_timing, expected_syllable in zip(char_timings, expected_syllables):
                char = char_timing['char']
                tone, pinyin_base = self.get_tone_from_pypinyin(char)
                
                boundaries.append({
                    'syllable': expected_syllable,
                    'character': char,
                    'pinyin_from_char': pinyin_base,
                    'predicted_tone': tone,
                    'start_time': char_timing['start_time'],
                    'end_time': char_timing['end_time'],
                    'duration': char_timing['duration'],
                    'method': 'guided_pypinyin'
                })
        
        elif len(char_timings) > len(expected_syllables):
            # More characters than expected syllables - group characters
            chars_per_syllable = len(char_timings) // len(expected_syllables)
            
            for i, expected_syllable in enumerate(expected_syllables):
                start_idx = i * chars_per_syllable
                end_idx = (i + 1) * chars_per_syllable if i < len(expected_syllables) - 1 else len(char_timings)
                
                # Get timing from first and last character in group
                start_time = char_timings[start_idx]['start_time']
                end_time = char_timings[end_idx - 1]['end_time']
                
                # Get characters in this group
                chars_in_group = [char_timings[j]['char'] for j in range(start_idx, end_idx)]
                main_char = chars_in_group[0]  # Use first character for tone lookup
                
                tone, pinyin_base = self.get_tone_from_pypinyin(main_char)
                
                boundaries.append({
                    'syllable': expected_syllable,
                    'character': ''.join(chars_in_group),
                    'main_character': main_char,
                    'pinyin_from_char': pinyin_base,
                    'predicted_tone': tone,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'method': 'grouped_pypinyin'
                })
        
        else:
            # Fewer characters than expected syllables - distribute proportionally
            if char_timings:
                total_start = char_timings[0]['start_time']
                total_end = char_timings[-1]['end_time']
                total_duration = total_end - total_start
                syllable_duration = total_duration / len(expected_syllables)
                
                for i, expected_syllable in enumerate(expected_syllables):
                    start_time = total_start + i * syllable_duration
                    end_time = start_time + syllable_duration
                    
                    # Find closest character for this time range
                    mid_time = (start_time + end_time) / 2
                    closest_char_idx = min(range(len(char_timings)),
                                         key=lambda j: abs(char_timings[j]['start_time'] - mid_time))
                    closest_char = char_timings[closest_char_idx]['char']
                    
                    tone, pinyin_base = self.get_tone_from_pypinyin(closest_char)
                    
                    boundaries.append({
                        'syllable': expected_syllable,
                        'character': closest_char,
                        'pinyin_from_char': pinyin_base,
                        'predicted_tone': tone,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': syllable_duration,
                        'method': 'distributed_pypinyin'
                    })
        
        return boundaries
    
    def predict_tones(self, audio_path, expected_syllables=None, show_alternatives=False):
        """Main function: detect characters and classify tones"""
        
        print(f"Processing: {os.path.basename(audio_path)}")
        if expected_syllables:
            print(f"Expected syllables: {expected_syllables}")
        
        # Step 1: Detect characters with timing
        transcription, char_timings = self.detect_characters_with_timing(audio_path)
        
        if not char_timings:
            print("No Chinese characters detected")
            return None
        
        print(f"Detected characters: {[ct['char'] for ct in char_timings]}")
        
        # Step 2: Classify tones
        if expected_syllables:
            # Use expected syllables to guide segmentation
            results = self.segment_with_expected_syllables(char_timings, expected_syllables)
        else:
            # Direct character-to-tone mapping
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
    
    def interactive_correction(self, results):
        """Allow user to correct misrecognized characters"""
        print("\nInteractive correction mode:")
        print("If any characters were misrecognized, you can correct them.")
        print("Press Enter to skip correction for a character.")
        
        corrected_results = []
        
        for result in results:
            char = result['character']
            current_tone = result['predicted_tone']
            
            print(f"\nCharacter: {char}")
            if current_tone is not None:
                tone_str = f"T{current_tone}" if current_tone > 0 else "Neutral"
                print(f"Current tone: {tone_str}")
            else:
                print("Current tone: UNKNOWN")
            
            # Show alternatives if available
            if 'alternatives' in result:
                print("Available pronunciations:")
                for i, alt in enumerate(result['alternatives']):
                    alt_tone = alt['tone']
                    tone_str = f"T{alt_tone}" if alt_tone > 0 else "Neutral"
                    print(f"  {i+1}. {alt['full_pinyin']} ({tone_str})")
            
            correction = input("Enter correct character (or number for alternative, or Enter to keep): ").strip()
            
            if correction:
                if correction.isdigit() and 'alternatives' in result:
                    # User selected an alternative
                    alt_idx = int(correction) - 1
                    if 0 <= alt_idx < len(result['alternatives']):
                        alt = result['alternatives'][alt_idx]
                        result['character'] = char  # Keep original character
                        result['pinyin'] = alt['pinyin']
                        result['predicted_tone'] = alt['tone']
                        result['method'] = 'user_selected_alternative'
                        print(f"Selected: {alt['full_pinyin']}")
                elif len(correction) == 1 and self.is_chinese_character(correction):
                    # User provided a different character
                    new_tone, new_pinyin = self.get_tone_from_pypinyin(correction)
                    result['character'] = correction
                    result['pinyin'] = new_pinyin
                    result['predicted_tone'] = new_tone
                    result['method'] = 'user_corrected'
                    print(f"Corrected to: {correction}")
            
            corrected_results.append(result)
        
        return corrected_results

def main():
    """Test the complete system"""
    
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
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Process audio file with expected syllables")
        print("2. Process audio file (auto-detection)")
        print("3. Test pypinyin with text input")
        print("4. Quit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '4':
            break
        elif choice == '1':
            audio_path = input("Audio file path: ").strip()
            syllables = input("Expected syllables (space-separated): ").strip()
            
            if os.path.exists(audio_path):
                expected_syllables = syllables.split() if syllables else None
                results = classifier.predict_tones(audio_path, expected_syllables, show_alternatives=True)
                
                if results:
                    print("\n" + classifier.format_results(results))
                    
                    # Option for interactive correction
                    correct = input("\nDo you want to correct any characters? (y/n): ").strip().lower()
                    if correct == 'y':
                        corrected_results = classifier.interactive_correction(results)
                        print("\nCorrected results:")
                        print(classifier.format_results(corrected_results, show_details=False))
                else:
                    print("No results obtained")
            else:
                print("Audio file not found")
        
        elif choice == '2':
            audio_path = input("Audio file path: ").strip()
            
            if os.path.exists(audio_path):
                results = classifier.predict_tones(audio_path, None, show_alternatives=True)
                
                if results:
                    print("\n" + classifier.format_results(results))
                else:
                    print("No results obtained")
            else:
                print("Audio file not found")
        
        elif choice == '3':
            text = input("Enter Chinese text: ").strip()
            if text:
                print("\nPypinyin results:")
                for char in text:
                    if classifier.is_chinese_character(char):
                        tone, pinyin = classifier.get_tone_from_pypinyin(char)
                        alternatives = classifier.get_multiple_pronunciations(char)
                        
                        print(f"Character: {char}")
                        if tone is not None:
                            tone_str = f"T{tone}" if tone > 0 else "Neutral"
                            print(f"  Main: {pinyin}{tone} ({tone_str})")
                        
                        if len(alternatives) > 1:
                            print(f"  Alternatives: {', '.join([alt['full_pinyin'] for alt in alternatives])}")
                        print()

if __name__ == "__main__":
    main()