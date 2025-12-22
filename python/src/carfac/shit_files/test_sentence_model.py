import os
import numpy as np
import librosa
import tensorflow as tf
import json
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("pypinyin not found. Install with: pip install pypinyin")

# Model paths
MODEL_DIR = r"C:\Users\maruk\carfac-SAI\python\src\carfac\modules\tone_detection_model\tone_detection_model_20250928_113915"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
MODEL_FILE = os.path.join(MODEL_DIR, "tone_detection.keras")

class SentenceToneClassifier:
    """Sentence-level tone classification with full context awareness"""
    
    def __init__(self, 
                 tone_model_path=MODEL_FILE,
                 tone_config_path=CONFIG_PATH,
                 wav2vec2_model="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"):
        
        print("=" * 80)
        print("SENTENCE-LEVEL TONE CLASSIFIER")
        print("Full sentence context-aware tone detection")
        print("=" * 80)
        
        # Load tone detection model
        print("\n[1/2] Loading trained tone detection model...")
        self.tone_model = tf.keras.models.load_model(tone_model_path)
        
        with open(tone_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.syllable_to_idx = self.config['syllable_to_idx']
        self.context_size = self.config['context_size']
        self.mel_input_shape = tuple(self.config['mel_input_shape'])
        self.num_classes = self.config['num_classes']
        
        print(f"✓ Tone model loaded!")
        print(f"  Context size: {self.context_size}")
        
        # Load wav2vec2
        print("\n[2/2] Loading wav2vec2 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()
        
        self.wav2vec2_vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.wav2vec2_vocab.items()}
        
        print(f"✓ Ready on {self.device}!")
    
    def load_audio(self, audio_input):
        """Load audio from file or array"""
        if isinstance(audio_input, str):
            audio_16k, _ = librosa.load(audio_input, sr=16000)
            audio_22k, _ = librosa.load(audio_input, sr=22050)
            return audio_16k, audio_22k
        else:
            raise ValueError("Only file paths supported")
    
    def detect_sentence_structure(self, audio_16k):
        """Detect all characters in sentence with timing"""
        audio_norm = librosa.util.normalize(audio_16k)
        
        inputs = self.wav2vec2_processor(audio_norm, sampling_rate=16000, 
                                         return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.wav2vec2_model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
        
        # Extract character timings
        char_timings = self._extract_timings(predicted_ids)
        
        return transcription, char_timings
    
    def _extract_timings(self, predicted_ids):
        """Extract character-level timing"""
        frames = predicted_ids.cpu().numpy()[0]
        frame_dur = 0.02
        
        timings = []
        current_char = None
        char_start = 0
        
        for i, token_id in enumerate(frames):
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                if token in ['<pad>', '<s>', '</s>', '<unk>', '|', '[PAD]', ' ', '']:
                    continue
                
                if token != current_char and token.strip() and self._is_chinese(token):
                    if current_char and current_char.strip():
                        dur = (i - char_start) * frame_dur
                        if dur > 0.02:
                            timings.append({
                                'char': current_char,
                                'start': char_start * frame_dur,
                                'end': i * frame_dur,
                                'duration': dur
                            })
                    current_char = token
                    char_start = i
        
        # Last character
        if current_char and current_char.strip():
            dur = (len(frames) - char_start) * frame_dur
            if dur > 0.02:
                timings.append({
                    'char': current_char,
                    'start': char_start * frame_dur,
                    'end': len(frames) * frame_dur,
                    'duration': dur
                })
        
        return timings
    
    def _is_chinese(self, char):
        """Check if Chinese character"""
        if not char:
            return False
        for c in char:
            if not ('\u4e00' <= c <= '\u9fff' or
                   '\u3400' <= c <= '\u4dbf' or
                   '\uf900' <= c <= '\ufaff'):
                return False
        return True
    
    def extract_sentence_features(self, audio_22k, char_timings):
        """Extract acoustic features for entire sentence"""
        sentence_features = []
        
        for idx, timing in enumerate(char_timings):
            # Extract character audio
            sr = 22050
            start_sample = int(timing['start'] * sr)
            end_sample = int(timing['end'] * sr)
            
            # Add padding
            pad = int(0.05 * sr)
            start_sample = max(0, start_sample - pad)
            end_sample = min(len(audio_22k), end_sample + pad)
            
            char_audio = audio_22k[start_sample:end_sample]
            
            # Get pinyin
            pinyin_result = self._get_pinyin(timing['char'])
            
            # Debug info
            print(f"  [{idx+1}] {timing['char']} ({pinyin_result['pinyin']})")
            print(f"      Duration: {timing['duration']:.3f}s, Audio samples: {len(char_audio)}")
            
            # Extract mel-spectrogram
            mel_spec = self._extract_mel(char_audio)
            
            if mel_spec is not None:
                print(f"      ✓ Mel shape: {mel_spec.shape}")
                sentence_features.append({
                    'char': timing['char'],
                    'pinyin': pinyin_result['pinyin'],
                    'syllable_id': self.syllable_to_idx.get(pinyin_result['pinyin'], 0),
                    'mel_spec': mel_spec,
                    'start': timing['start'],
                    'end': timing['end'],
                    'duration': timing['duration'],
                    'dict_tone': pinyin_result['tone']
                })
            else:
                print(f"      ✗ Mel extraction failed (audio too short?)")
        
        return sentence_features
    
    def _get_pinyin(self, char):
        """Get pinyin from character"""
        if not PYPINYIN_AVAILABLE:
            return {'pinyin': 'unknown', 'tone': None}
        
        try:
            result = pinyin(char, style=Style.TONE3, strict=False)
            if result and result[0]:
                py_tone = result[0][0]
                if py_tone and py_tone[-1].isdigit():
                    tone = int(py_tone[-1])
                    if tone == 5:
                        tone = 0
                    return {'pinyin': py_tone[:-1], 'tone': tone}
                return {'pinyin': py_tone, 'tone': 0}
            return {'pinyin': 'unknown', 'tone': None}
        except:
            return {'pinyin': 'unknown', 'tone': None}
    
    def _extract_mel(self, audio):
        """Extract mel-spectrogram"""
        try:
            sr = 22050
            
            # More lenient minimum length (was 0.5s, now 0.1s)
            if len(audio) < 0.1 * sr:
                return None
            
            y, _ = librosa.effects.trim(audio, top_db=20)
            
            # Skip if trimmed audio is too short
            if len(y) < 0.05 * sr:
                return None
                
            y = librosa.util.normalize(y)
            
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=64, fmin=72, fmax=504,
                hop_length=int(sr * 0.013), n_fft=int(sr * 0.025),
                window='hann'
            )
            
            return librosa.power_to_db(mel, ref=np.max).T
        except Exception as e:
            print(f"        Mel extraction error: {e}")
            return None
    
    def predict_sentence_tones(self, sentence_features):
        """Predict tones for entire sentence with full context"""
        if not sentence_features:
            return []
        
        # Create tri-tone segments for all characters
        all_segments = []
        for feat in sentence_features:
            segments = self._create_segments(feat)
            all_segments.extend(segments)
        
        # Create context windows for each segment
        predictions = []
        context_len = self.context_size * 2 + 1
        
        for i, feat in enumerate(sentence_features):
            # Get context indices
            char_segment_start = i * 3
            char_segment_end = char_segment_start + 3
            
            char_predictions = []
            
            for seg_idx in range(char_segment_start, char_segment_end):
                # Build context window
                context_segs = []
                context_durs = []
                context_syll_ids = []
                
                for offset in range(-self.context_size, self.context_size + 1):
                    ctx_idx = seg_idx + offset
                    
                    if 0 <= ctx_idx < len(all_segments):
                        seg = all_segments[ctx_idx]
                    elif ctx_idx < 0:
                        seg = all_segments[0]
                    else:
                        seg = all_segments[-1]
                    
                    context_segs.append(seg['segment'])
                    context_durs.append(seg['duration'])
                    context_syll_ids.append(seg['syllable_id'])
                
                # Prepare model input
                model_input = self._prepare_input(context_segs, context_durs, context_syll_ids)
                
                # Predict
                probs = self.tone_model.predict(model_input, verbose=0)
                char_predictions.append(probs[0])
            
            # Average predictions across 3 segments
            avg_probs = np.mean(char_predictions, axis=0)
            pred_tone = np.argmax(avg_probs) + 1
            confidence = np.max(avg_probs)
            
            predictions.append({
                'char': feat['char'],
                'pinyin': feat['pinyin'],
                'dict_tone': feat['dict_tone'],
                'pred_tone': pred_tone,
                'confidence': confidence,
                'probabilities': {f'T{i+1}': float(p) for i, p in enumerate(avg_probs)},
                'start': feat['start'],
                'end': feat['end'],
                'duration': feat['duration']
            })
        
        return predictions
    
    def _create_segments(self, feat):
        """Create tri-tone segments"""
        mel = feat['mel_spec']
        total_frames = mel.shape[0]
        seg_len = total_frames // 3
        
        segments = []
        for i in range(3):
            start = max(0, i * seg_len - seg_len // 2)
            end = min(total_frames, (i + 1) * seg_len + seg_len // 2)
            
            if end > start:
                segments.append({
                    'segment': mel[start:end, :],
                    'syllable_id': feat['syllable_id'],
                    'duration': (end - start) / (22050 / (22050 * 0.013))
                })
        
        return segments
    
    def _prepare_input(self, context_segs, context_durs, context_syll_ids):
        """Prepare input for model"""
        context_len = self.context_size * 2 + 1
        max_len = 150
        
        model_inputs = []
        
        # Mel inputs
        for seg in context_segs:
            if seg.shape[0] > max_len:
                seg = seg[:max_len, :]
            elif seg.shape[0] < max_len:
                pad = np.zeros((max_len - seg.shape[0], seg.shape[1]))
                seg = np.vstack([seg, pad])
            model_inputs.append(np.expand_dims(seg, axis=0))
        
        # Duration and syllable inputs
        model_inputs.append(np.expand_dims(context_durs, axis=0))
        model_inputs.append(np.expand_dims(context_syll_ids, axis=0))
        
        return model_inputs
    
    def analyze_sentence(self, audio_path):
        """Complete sentence analysis"""
        print(f"\nAnalyzing: {os.path.basename(audio_path)}")
        
        # Load audio
        audio_16k, audio_22k = self.load_audio(audio_path)
        
        print("\n[Step 1/3] Detecting sentence structure...")
        transcription, char_timings = self.detect_sentence_structure(audio_16k)
        print(f"  Transcription: {transcription}")
        print(f"  Characters: {[t['char'] for t in char_timings]}")
        
        if not char_timings:
            return None
        
        print("\n[Step 2/3] Extracting acoustic features...")
        sentence_features = self.extract_sentence_features(audio_22k, char_timings)
        print(f"  Extracted features for {len(sentence_features)} characters")
        
        print("\n[Step 3/3] Predicting tones with full sentence context...")
        predictions = self.predict_sentence_tones(sentence_features)
        
        return {
            'transcription': transcription,
            'audio_path': audio_path,
            'predictions': predictions
        }
    
    def format_results(self, analysis):
        """Format results"""
        if not analysis:
            return "No analysis available"
        
        output = []
        output.append("\n" + "=" * 80)
        output.append("SENTENCE-LEVEL TONE ANALYSIS")
        output.append("=" * 80)
        output.append(f"File: {os.path.basename(analysis['audio_path'])}")
        output.append(f"Sentence: {analysis['transcription']}")
        output.append(f"Length: {len(analysis['predictions'])} characters")
        output.append("")
        
        # Character details
        output.append("CHARACTER ANALYSIS:")
        output.append("-" * 80)
        
        for i, pred in enumerate(analysis['predictions']):
            output.append(f"\n[{i+1}] {pred['char']} ({pred['pinyin']})")
            output.append(f"    Time: {pred['start']:.2f}s - {pred['end']:.2f}s")
            
            if pred['dict_tone'] is not None:
                tone_names = {0: "Neutral", 1: "T1", 2: "T2", 3: "T3", 4: "T4"}
                output.append(f"    Dictionary: {tone_names.get(pred['dict_tone'], '?')}")
            
            output.append(f"    Predicted: T{pred['pred_tone']} (conf: {pred['confidence']:.3f})")
            output.append(f"    Probs: " + ", ".join([f"{k}:{v:.2f}" for k, v in pred['probabilities'].items()]))
        
        # Summary
        output.append("\n" + "=" * 80)
        output.append("SUMMARY:")
        chars = ''.join([p['char'] for p in analysis['predictions']])
        pinyins = ' '.join([p['pinyin'] for p in analysis['predictions']])
        dict_tones = ' '.join([str(p['dict_tone']) if p['dict_tone'] is not None else '?' 
                               for p in analysis['predictions']])
        pred_tones = ' '.join([str(p['pred_tone']) for p in analysis['predictions']])
        
        output.append(f"  Characters: {chars}")
        output.append(f"  Pinyin: {pinyins}")
        output.append(f"  Dictionary: {dict_tones}")
        output.append(f"  Predicted: {pred_tones}")
        
        # Agreement
        agreements = sum(1 for p in analysis['predictions'] 
                        if p['dict_tone'] == p['pred_tone'] and p['dict_tone'] is not None)
        comparisons = sum(1 for p in analysis['predictions'] if p['dict_tone'] is not None)
        
        if comparisons > 0:
            output.append(f"  Agreement: {agreements}/{comparisons} ({100*agreements/comparisons:.1f}%)")
        
        output.append("=" * 80)
        
        return "\n".join(output)

def main():
    """Main loop"""
    print("SENTENCE-LEVEL TONE CLASSIFIER")
    print("=" * 80)
    
    try:
        classifier = SentenceToneClassifier()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    print("\nReady! Type 'quit' to exit")
    
    while True:
        audio_path = input("\nEnter audio file path: ").strip().strip('"').strip("'")
        
        if audio_path.lower() in ['quit', 'exit', 'q', '']:
            print("Goodbye!")
            break
        
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue
        
        try:
            analysis = classifier.analyze_sentence(audio_path)
            if analysis:
                print(classifier.format_results(analysis))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()