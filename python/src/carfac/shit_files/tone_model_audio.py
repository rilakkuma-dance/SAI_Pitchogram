import os
import numpy as np
import librosa
import tensorflow as tf
import json
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datetime import datetime

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("pypinyin not found. Install with: pip install pypinyin")

# Model paths
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
# Model paths - use absolute path
MODEL_DIR = r"C:\Users\maruk\carfac-SAI\python\src\carfac\modules\tone_detection_model\tone_detection_model_20250928_113915"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
MODEL_FILE = os.path.join(MODEL_DIR, "tone_detection.keras")

class HybridToneClassifier:
    """Combines wav2vec2 character detection + your trained tone classifier"""
    
    def __init__(self, 
                 tone_model_path=MODEL_FILE,
                 tone_config_path=CONFIG_PATH,
                 wav2vec2_model="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"):
        """Initialize both models"""
        
        print("=" * 80)
        print("HYBRID TONE CLASSIFIER")
        print("Wav2Vec2 Character Detection + Your Trained Tone Model")
        print("=" * 80)
        
        # Load your tone detection model
        print("\n[1/2] Loading your trained tone detection model...")
        self.tone_model = tf.keras.models.load_model(tone_model_path)
        
        with open(tone_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.syllable_to_idx = self.config['syllable_to_idx']
        self.context_size = self.config['context_size']
        self.mel_input_shape = tuple(self.config['mel_input_shape'])
        self.num_classes = self.config['num_classes']
        
        print(f"✓ Tone model loaded!")
        print(f"  Context size: {self.context_size}")
        print(f"  Mel input shape: {self.mel_input_shape}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Syllable vocabulary size: {len(self.syllable_to_idx)}")
        
        # Load wav2vec2 for character detection
        print("\n[2/2] Loading wav2vec2 Chinese character model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()
        
        self.wav2vec2_vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.wav2vec2_vocab.items()}
        
        print(f"✓ Wav2vec2 model loaded on {self.device}!")
        print(f"  Vocabulary size: {len(self.wav2vec2_vocab)}")
        
        print("\n✓ Both models ready!")
    
    def load_audio(self, audio_input):
        """Load audio file"""
        try:
            if isinstance(audio_input, str):
                # Load for wav2vec2 (16kHz)
                audio_16k, _ = librosa.load(audio_input, sr=16000)
                # Load for tone model (22.05kHz)
                audio_22k, _ = librosa.load(audio_input, sr=22050, duration=None)
                
                print(f"\nLoaded: {os.path.basename(audio_input)}")
                print(f"Duration: {len(audio_16k)/16000:.2f}s")
                
                return audio_16k, audio_22k
            else:
                raise ValueError("Only file paths supported for now")
                
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def detect_characters(self, audio_16k):
        """Detect Chinese characters using wav2vec2"""
        try:
            audio_normalized = librosa.util.normalize(audio_16k)
            
            inputs = self.wav2vec2_processor(
                audio_normalized, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
            
            print(f"Transcription: '{transcription}'")
            
            # Extract character timings
            char_timings = self.extract_character_timings(predicted_ids)
            
            return transcription, char_timings
            
        except Exception as e:
            print(f"Character detection error: {e}")
            return "", []
    
    def extract_character_timings(self, predicted_ids):
        """Extract character-level timing information"""
        frame_predictions = predicted_ids.cpu().numpy()[0]
        frame_duration = 0.02  # 50Hz
        
        char_timings = []
        current_char = None
        char_start = 0
        
        for i, token_id in enumerate(frame_predictions):
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                if token in ['<pad>', '<s>', '</s>', '<unk>', '|', '[PAD]', ' ', '']:
                    continue
                
                if token != current_char and token.strip() and self.is_chinese_character(token):
                    if current_char is not None and current_char.strip():
                        duration = (i - char_start) * frame_duration
                        if duration > 0.02:
                            char_timings.append({
                                'char': current_char,
                                'start_time': char_start * frame_duration,
                                'end_time': i * frame_duration,
                                'duration': duration
                            })
                    
                    current_char = token
                    char_start = i
        
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
        for c in char:
            if not ('\u4e00' <= c <= '\u9fff' or
                   '\u3400' <= c <= '\u4dbf' or
                   '\uf900' <= c <= '\ufaff'):
                return False
        return True
    
    def extract_character_audio(self, audio_22k, start_time, end_time):
        """Extract audio segment for a specific character"""
        sr = 22050
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Add small padding
        padding = int(0.05 * sr)  # 50ms padding
        start_sample = max(0, start_sample - padding)
        end_sample = min(len(audio_22k), end_sample + padding)
        
        char_audio = audio_22k[start_sample:end_sample]
        return char_audio
    
    def extract_mel_spectrogram(self, audio_segment):
        """Extract mel-spectrogram using your model's method"""
        try:
            sr = 22050
            y = audio_segment
            
            if len(y) < 0.5 * sr:
                return None
            
            # Trim and normalize
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=64,
                fmin=72,
                fmax=504,
                hop_length=int(sr * 0.013),
                n_fft=int(sr * 0.025),
                window='hann'
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec.T
            
        except Exception as e:
            print(f"Error extracting mel-spectrogram: {e}")
            return None
    
    def create_tri_tone_segments(self, mel_spec, syllable_info):
        """Create tri-tone segments"""
        if mel_spec is None or len(mel_spec) == 0:
            return []
        
        total_frames = mel_spec.shape[0]
        segment_length = total_frames // 3
        
        segments = []
        
        for i in range(3):
            start_frame = max(0, i * segment_length - segment_length // 2)
            end_frame = min(total_frames, (i + 1) * segment_length + segment_length // 2)
            
            if end_frame > start_frame:
                segment = mel_spec[start_frame:end_frame, :]
                
                segments.append({
                    'segment': segment,
                    'syllable': syllable_info['syllable'],
                    'syllable_id': syllable_info['syllable_id'],
                    'duration': (end_frame - start_frame) / (22050 / (22050 * 0.013))
                })
        
        return segments
    
    def create_context_features(self, segments):
        """Create contextual features"""
        if not segments:
            return []
        
        contextual_data = []
        context_length = self.context_size * 2 + 1
        
        for center_idx in range(len(segments)):
            context_segments = []
            context_durations = []
            context_syllable_ids = []
            
            for offset in range(-self.context_size, self.context_size + 1):
                idx = center_idx + offset
                
                if 0 <= idx < len(segments):
                    context_segments.append(segments[idx]['segment'])
                    context_durations.append(segments[idx]['duration'])
                    context_syllable_ids.append(segments[idx]['syllable_id'])
                else:
                    if idx < 0:
                        context_segments.append(segments[0]['segment'])
                        context_durations.append(segments[0]['duration'])
                        context_syllable_ids.append(segments[0]['syllable_id'])
                    else:
                        context_segments.append(segments[-1]['segment'])
                        context_durations.append(segments[-1]['duration'])
                        context_syllable_ids.append(segments[-1]['syllable_id'])
            
            contextual_data.append({
                'context_segments': context_segments,
                'context_durations': context_durations,
                'context_syllable_ids': context_syllable_ids,
                'target_syllable': segments[center_idx]['syllable']
            })
        
        return contextual_data
    
    def preprocess_for_prediction(self, mel_spec, syllable_text):
        """Preprocess for your tone model"""
        syllable_id = self.syllable_to_idx.get(syllable_text, 0)
        syllable_info = {
            'syllable': syllable_text,
            'syllable_id': syllable_id
        }
        
        segments = self.create_tri_tone_segments(mel_spec, syllable_info)
        if not segments:
            return None
        
        contextual_data = self.create_context_features(segments)
        if not contextual_data:
            return None
        
        context_length = self.context_size * 2 + 1
        max_length = 150
        
        processed_samples = []
        
        for ctx_data in contextual_data:
            mel_contexts = []
            for seg in ctx_data['context_segments']:
                if seg.shape[0] > max_length:
                    seg = seg[:max_length, :]
                elif seg.shape[0] < max_length:
                    padding = np.zeros((max_length - seg.shape[0], seg.shape[1]))
                    seg = np.vstack([seg, padding])
                mel_contexts.append(seg)
            
            while len(mel_contexts) < context_length:
                mel_contexts.append(np.zeros((max_length, 64)))
            mel_contexts = mel_contexts[:context_length]
            
            durations = ctx_data['context_durations'][:]
            while len(durations) < context_length:
                durations.append(0.0)
            durations = durations[:context_length]
            
            syllable_ids = ctx_data['context_syllable_ids'][:]
            while len(syllable_ids) < context_length:
                syllable_ids.append(0)
            syllable_ids = syllable_ids[:context_length]
            
            processed_samples.append({
                'mel_contexts': mel_contexts,
                'durations': durations,
                'syllable_ids': syllable_ids
            })
        
        return processed_samples
    
    def predict_character_tone(self, audio_segment, syllable_text):
        """Predict tone for a single character using your model"""
        try:
            # Extract mel-spectrogram
            mel_spec = self.extract_mel_spectrogram(audio_segment)
            if mel_spec is None:
                return None
            
            # Preprocess
            processed_samples = self.preprocess_for_prediction(mel_spec, syllable_text)
            if not processed_samples:
                return None
            
            # Prepare model inputs
            context_length = self.context_size * 2 + 1
            sample = processed_samples[0]
            
            model_inputs = []
            
            # Mel-spectrogram inputs
            for i in range(context_length):
                mel_input = np.expand_dims(sample['mel_contexts'][i], axis=0)
                model_inputs.append(mel_input)
            
            # Duration and syllable inputs
            duration_input = np.expand_dims(sample['durations'], axis=0)
            syllable_input = np.expand_dims(sample['syllable_ids'], axis=0)
            model_inputs.extend([duration_input, syllable_input])
            
            # Make prediction
            prediction_probs = self.tone_model.predict(model_inputs, verbose=0)
            predicted_class = np.argmax(prediction_probs[0])
            confidence = np.max(prediction_probs[0])
            
            predicted_tone = predicted_class + 1
            
            return {
                'predicted_tone': predicted_tone,
                'confidence': confidence,
                'probabilities': {f'T{i+1}': float(prob) for i, prob in enumerate(prediction_probs[0])}
            }
            
        except Exception as e:
            print(f"Error predicting tone: {e}")
            return None
    
    def get_pinyin_from_char(self, char):
        """Get pinyin using pypinyin"""
        if not PYPINYIN_AVAILABLE:
            return None, None
        
        try:
            result = pinyin(char, style=Style.TONE3, strict=False)
            if result and result[0]:
                pinyin_with_tone = result[0][0]
                if pinyin_with_tone and pinyin_with_tone[-1].isdigit():
                    tone = int(pinyin_with_tone[-1])
                    if tone == 5:
                        tone = 0
                    return tone, pinyin_with_tone[:-1]
                else:
                    return 0, pinyin_with_tone
            return None, None
        except:
            return None, None
    
    def analyze_audio(self, audio_path):
        """Complete analysis pipeline"""
        
        # Load audio
        audio_16k, audio_22k = self.load_audio(audio_path)
        if audio_16k is None or audio_22k is None:
            return None
        
        print("\n" + "=" * 80)
        print("[Step 1/3] Detecting Chinese characters with wav2vec2...")
        print("=" * 80)
        
        transcription, char_timings = self.detect_characters(audio_16k)
        
        if not char_timings:
            print("❌ No Chinese characters detected")
            return None
        
        print(f"✓ Detected {len(char_timings)} characters: {[c['char'] for c in char_timings]}")
        
        print("\n" + "=" * 80)
        print("[Step 2/3] Getting pinyin for each character...")
        print("=" * 80)
        
        # Get pinyin for each character
        for char_timing in char_timings:
            pypinyin_tone, pinyin_base = self.get_pinyin_from_char(char_timing['char'])
            char_timing['pinyin'] = pinyin_base if pinyin_base else 'unknown'
            char_timing['pypinyin_tone'] = pypinyin_tone
            print(f"  {char_timing['char']} → {char_timing['pinyin']}")
        
        print("\n" + "=" * 80)
        print("[Step 3/3] Classifying tones with your trained model...")
        print("=" * 80)
        
        # Analyze each character
        results = []
        for idx, char_timing in enumerate(char_timings):
            char = char_timing['char']
            pinyin_text = char_timing['pinyin']
            
            print(f"\nAnalyzing character {idx+1}/{len(char_timings)}: {char} ({pinyin_text})")
            
            # Extract audio for this character
            char_audio = self.extract_character_audio(
                audio_22k, 
                char_timing['start_time'], 
                char_timing['end_time']
            )
            
            # Predict tone
            tone_result = self.predict_character_tone(char_audio, pinyin_text)
            
            if tone_result:
                result = {
                    'char_idx': idx,
                    'character': char,
                    'pinyin': pinyin_text,
                    'pypinyin_tone': char_timing['pypinyin_tone'],
                    'model_tone': tone_result['predicted_tone'],
                    'confidence': tone_result['confidence'],
                    'probabilities': tone_result['probabilities'],
                    'start_time': char_timing['start_time'],
                    'end_time': char_timing['end_time'],
                    'duration': char_timing['duration']
                }
                print(f"  → Model predicts: T{tone_result['predicted_tone']} (confidence: {tone_result['confidence']:.3f})")
            else:
                result = {
                    'char_idx': idx,
                    'character': char,
                    'pinyin': pinyin_text,
                    'pypinyin_tone': char_timing['pypinyin_tone'],
                    'model_tone': None,
                    'confidence': 0.0,
                    'probabilities': {},
                    'start_time': char_timing['start_time'],
                    'end_time': char_timing['end_time'],
                    'duration': char_timing['duration']
                }
                print(f"  → Failed to predict")
            
            results.append(result)
        
        return {
            'transcription': transcription,
            'total_characters': len(char_timings),
            'results': results,
            'audio_path': audio_path
        }
    
    def format_results(self, analysis):
        """Format results for display"""
        if not analysis:
            return "No analysis available"
        
        output = []
        output.append("\n" + "=" * 80)
        output.append("HYBRID TONE CLASSIFICATION RESULTS")
        output.append("=" * 80)
        output.append(f"Audio File: {os.path.basename(analysis['audio_path'])}")
        output.append(f"Transcription: {analysis['transcription']}")
        output.append(f"Total Characters: {analysis['total_characters']}")
        output.append("")
        
        output.append("CHARACTER-BY-CHARACTER ANALYSIS:")
        output.append("-" * 80)
        
        for result in analysis['results']:
            char_num = result['char_idx'] + 1
            output.append(f"\n[{char_num}] Character: {result['character']}")
            output.append(f"    Pinyin: {result['pinyin']}")
            output.append(f"    Time: {result['start_time']:.3f}s - {result['end_time']:.3f}s (duration: {result['duration']:.3f}s)")
            
            # Dictionary tone
            if result['pypinyin_tone'] is not None:
                tone_names = {0: "Neutral", 1: "T1 (High)", 2: "T2 (Rising)", 
                             3: "T3 (Dip)", 4: "T4 (Falling)"}
                output.append(f"    Dictionary Tone: {tone_names.get(result['pypinyin_tone'], 'Unknown')}")
            else:
                output.append(f"    Dictionary Tone: Unknown")
            
            # Model prediction
            if result['model_tone']:
                tone_names = {1: "T1 (High)", 2: "T2 (Rising)", 3: "T3 (Dip)", 4: "T4 (Falling)"}
                output.append(f"    Model Prediction: {tone_names.get(result['model_tone'], 'Unknown')}")
                output.append(f"    Confidence: {result['confidence']:.3f}")
                output.append(f"    Probabilities:")
                for tone, prob in result['probabilities'].items():
                    output.append(f"      {tone}: {prob:.3f}")
            else:
                output.append(f"    Model Prediction: Failed")
        
        # Summary
        output.append("\n" + "=" * 80)
        output.append("SUMMARY:")
        
        chars = [r['character'] for r in analysis['results']]
        pinyins = [r['pinyin'] for r in analysis['results']]
        dict_tones = [str(r['pypinyin_tone']) if r['pypinyin_tone'] is not None else '?' 
                     for r in analysis['results']]
        model_tones = [str(r['model_tone']) if r['model_tone'] else '?' 
                      for r in analysis['results']]
        
        output.append(f"  Characters: {''.join(chars)}")
        output.append(f"  Pinyin: {' '.join(pinyins)}")
        output.append(f"  Dictionary Tones: {' '.join(dict_tones)}")
        output.append(f"  Model Tones: {' '.join(model_tones)}")
        
        # Agreement
        agreements = 0
        comparisons = 0
        for r in analysis['results']:
            if r['pypinyin_tone'] is not None and r['model_tone'] is not None:
                comparisons += 1
                if r['pypinyin_tone'] == r['model_tone']:
                    agreements += 1
        
        if comparisons > 0:
            agreement_pct = (agreements / comparisons) * 100
            output.append(f"  Agreement: {agreements}/{comparisons} ({agreement_pct:.1f}%)")
        
        output.append("=" * 80)
        
        return "\n".join(output)

def main():
    """Main testing loop"""
    
    print("=" * 80)
    print("HYBRID TONE CLASSIFIER")
    print("Wav2Vec2 Character Detection + Your Trained Tone Model")
    print("=" * 80)
    
    # Initialize classifier
    try:
        classifier = HybridToneClassifier()
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Ready to analyze audio files!")
    print("Type 'quit', 'exit', or 'q' to end")
    print("=" * 80)
    
    # Main loop
    while True:
        print("\n")
        audio_path = input("Enter audio file path (or 'quit' to exit): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q', '']:
            print("\nExiting program. Goodbye!")
            break
        
        audio_path = audio_path.strip('"').strip("'")
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            continue
        
        try:
            analysis = classifier.analyze_audio(audio_path)
            
            if analysis:
                print(classifier.format_results(analysis))
            else:
                print("❌ Analysis failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()