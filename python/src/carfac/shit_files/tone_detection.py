import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import os
from typing import List, Dict, Optional
import numpy as np
import librosa
import tensorflow as tf
import json
import glob
from tqdm import tqdm
import pandas as pd
from datetime import datetime

class ToneClassifierTester:
    """Test the trained tone classifier on new audio files"""
    
    def __init__(self, model_path, config_path):
        """Load the trained model and configuration"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.syllable_to_idx = self.config['syllable_to_idx']
        self.context_size = self.config['context_size']
        self.mel_input_shape = tuple(self.config['mel_input_shape'])
        self.num_classes = self.config['num_classes']
        
        print(f"Model loaded successfully!")
        print(f"Context size: {self.context_size}")
        print(f"Mel input shape: {self.mel_input_shape}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Syllable vocabulary size: {len(self.syllable_to_idx)}")
    
    def extract_mel_spectrogram(self, audio_path):
        """Extract mel-spectrogram using the same method as training"""
        try:
            # Use same parameters as training
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            
            if len(y) < 0.5 * sr:
                return None
            
            # Trim and normalize
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)
            
            # Extract mel-spectrogram with same parameters
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=64,
                fmin=72,    # Scaled frequency range
                fmax=504,
                hop_length=int(sr * 0.013),
                n_fft=int(sr * 0.025),
                window='hann'
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec.T
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def create_tri_tone_segments(self, mel_spec, syllable_info):
        """Create tri-tone segments using the same method as training"""
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
        """Create contextual features using the same method as training"""
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
    
    def preprocess_for_prediction(self, mel_spec, syllable_text="unknown"):
        """Preprocess audio for prediction"""
        # Create syllable info
        syllable_id = self.syllable_to_idx.get(syllable_text, 0)  # Default to 0 if unknown
        syllable_info = {
            'syllable': syllable_text,
            'syllable_id': syllable_id
        }
        
        # Create segments
        segments = self.create_tri_tone_segments(mel_spec, syllable_info)
        if not segments:
            return None
        
        # Create context features
        contextual_data = self.create_context_features(segments)
        if not contextual_data:
            return None
        
        # Process for model input
        context_length = self.context_size * 2 + 1
        max_length = 150  # Same as training
        
        processed_samples = []
        
        for ctx_data in contextual_data:
            # Process mel contexts
            mel_contexts = []
            for seg in ctx_data['context_segments']:
                if seg.shape[0] > max_length:
                    seg = seg[:max_length, :]
                elif seg.shape[0] < max_length:
                    padding = np.zeros((max_length - seg.shape[0], seg.shape[1]))
                    seg = np.vstack([seg, padding])
                mel_contexts.append(seg)
            
            # Ensure correct context length
            while len(mel_contexts) < context_length:
                mel_contexts.append(np.zeros((max_length, 64)))
            mel_contexts = mel_contexts[:context_length]
            
            # Process durations
            durations = ctx_data['context_durations'][:]
            while len(durations) < context_length:
                durations.append(0.0)
            durations = durations[:context_length]
            
            # Process syllable IDs
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
    
    def predict_tone(self, audio_path, syllable_text="unknown"):
        """Predict tone for a single audio file"""
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio_path)
        if mel_spec is None:
            return None
        
        # Preprocess
        processed_samples = self.preprocess_for_prediction(mel_spec, syllable_text)
        if not processed_samples:
            return None
        
        # Prepare inputs for model
        context_length = self.context_size * 2 + 1
        
        # Use the first processed sample (could average if multiple)
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
        prediction_probs = self.model.predict(model_inputs, verbose=0)
        predicted_class = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0])
        
        # Convert to tone number (1-4)
        predicted_tone = predicted_class + 1
        
        return {
            'predicted_tone': predicted_tone,
            'confidence': confidence,
            'probabilities': prediction_probs[0].tolist(),
            'class_probabilities': {f'T{i+1}': prob for i, prob in enumerate(prediction_probs[0])}
        }
    
    def test_directory(self, test_dir, output_file=None):
        """Test all audio files in a directory"""
        print(f"Testing audio files in: {test_dir}")
        
        # Find audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(test_dir, ext)))
            # Also search subdirectories
            audio_files.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))
        
        if not audio_files:
            print("No audio files found!")
            return None
        
        print(f"Found {len(audio_files)} audio files")
        
        results = []
        
        for audio_path in tqdm(audio_files, desc="Testing files"):
            filename = os.path.basename(audio_path)
            
            # Try to extract syllable from filename if possible
            syllable_text = "unknown"
            try:
                # Common naming patterns
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) > 0 and parts[0]:
                        # Remove tone number if present
                        word = parts[0]
                        if word[-1].isdigit():
                            syllable_text = word[:-1]
                        else:
                            syllable_text = word
            except:
                pass
            
            # Make prediction
            result = self.predict_tone(audio_path, syllable_text)
            
            if result is not None:
                result_entry = {
                    'filename': filename,
                    'filepath': audio_path,
                    'syllable_used': syllable_text,
                    'predicted_tone': result['predicted_tone'],
                    'confidence': result['confidence'],
                    'T1_prob': result['class_probabilities']['T1'],
                    'T2_prob': result['class_probabilities']['T2'],
                    'T3_prob': result['class_probabilities']['T3'],
                    'T4_prob': result['class_probabilities']['T4']
                }
                results.append(result_entry)
            else:
                results.append({
                    'filename': filename,
                    'filepath': audio_path,
                    'syllable_used': syllable_text,
                    'predicted_tone': 'ERROR',
                    'confidence': 0.0,
                    'T1_prob': 0.0,
                    'T2_prob': 0.0,
                    'T3_prob': 0.0,
                    'T4_prob': 0.0
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Print summary
        print(f"\nTesting Results Summary:")
        print(f"Total files processed: {len(results)}")
        
        if len(df[df['predicted_tone'] != 'ERROR']) > 0:
            valid_results = df[df['predicted_tone'] != 'ERROR']
            print(f"Successful predictions: {len(valid_results)}")
            print(f"Failed predictions: {len(df) - len(valid_results)}")
            
            print(f"\nTone Distribution:")
            tone_counts = valid_results['predicted_tone'].value_counts().sort_index()
            for tone, count in tone_counts.items():
                print(f"  T{tone}: {count} files")
            
            print(f"\nConfidence Statistics:")
            print(f"  Mean confidence: {valid_results['confidence'].mean():.3f}")
            print(f"  Min confidence: {valid_results['confidence'].min():.3f}")
            print(f"  Max confidence: {valid_results['confidence'].max():.3f}")
            
            # Show low confidence predictions
            low_conf = valid_results[valid_results['confidence'] < 0.7]
            if len(low_conf) > 0:
                print(f"\nLow Confidence Predictions (< 0.7):")
                for _, row in low_conf.iterrows():
                    print(f"  {row['filename']}: T{row['predicted_tone']} (conf: {row['confidence']:.3f})")
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"tone_predictions_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return df

def main():
    """Main testing function"""
    
    # Configuration
    model_dir = "corrected_paper_models/paper_corrected_20250928_113915"  # Update this path
    model_path = os.path.join(model_dir, "corrected_paper_model.keras")
    config_path = os.path.join(model_dir, "config.json")
    
    test_dir = r"C:\Users\maruk\carfac-SAI\python\src\carfac\reference"
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update the model_dir path to point to your trained model")
        return
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    print("=" * 60)
    print("Tone Classifier Testing")
    print("=" * 60)
    
    # Load model and test
    tester = ToneClassifierTester(model_path, config_path)
    
    # Test the directory
    results = tester.test_directory(test_dir)
    
    if results is not None:
        print("\nTesting completed successfully!")
        
        # Option to test individual files
        while True:
            test_single = input("\nTest a single file? (y/n): ").strip().lower()
            if test_single != 'y':
                break
            
            file_path = input("Enter audio file path: ").strip()
            if os.path.exists(file_path):
                result = tester.predict_tone(file_path)
                if result:
                    print(f"Prediction: T{result['predicted_tone']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    print("All probabilities:")
                    for tone, prob in result['class_probabilities'].items():
                        print(f"  {tone}: {prob:.3f}")
                else:
                    print("Failed to process file")
            else:
                print("File not found")
    else:
        print("Testing failed")

if __name__ == "__main__":
    main()

class ChineseCharToneLookup:
    """Direct tone lookup from Chinese characters detected by wav2vec2"""
    
    def __init__(self, wav2vec2_model="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"):
        """Initialize with Chinese wav2vec2 and tone dictionary"""
        
        # Load Chinese wav2vec2
        print("Loading Chinese wav2vec2...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()
        
        # Get vocabulary
        self.vocab = self.processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Chinese wav2vec2 loaded successfully")
        
        # Load Chinese character tone dictionary
        self.tone_dict = self.load_tone_dictionary()
        print(f"Loaded tone dictionary with {len(self.tone_dict)} characters")
    
    def load_tone_dictionary(self):
        """Load comprehensive Chinese character to tone mapping"""
        # This is a sample dictionary - in practice, you'd load from a comprehensive database
        tone_dict = {
            # Common characters with their tones
            '我': 3,    # wo3
            '你': 3,    # ni3  
            '他': 1,    # ta1
            '她': 1,    # ta1
            '它': 1,    # ta1
            '的': 5,    # de (neutral tone)
            '是': 4,    # shi4
            '有': 3,    # you3
            '在': 4,    # zai4
            '了': 5,    # le (neutral tone)
            '不': 4,    # bu4
            '人': 2,    # ren2
            '一': 1,    # yi1
            '二': 4,    # er4
            '三': 1,    # san1
            '四': 4,    # si4
            '五': 3,    # wu3
            '六': 4,    # liu4
            '七': 1,    # qi1
            '八': 1,    # ba1
            '九': 3,    # jiu3
            '十': 2,    # shi2
            
            # Your test words
            '矮': 3,    # ai3 (short)
            '爱': 4,    # ai4 (love) 
            '猫': 1,    # mao1 (cat)
            '毛': 2,    # mao2 (hair)
            '帽': 4,    # mao4 (hat)
            
            # Common words
            '好': 3,    # hao3
            '很': 3,    # hen3
            '大': 4,    # da4
            '小': 3,    # xiao3
            '多': 1,    # duo1
            '少': 3,    # shao3
            '来': 2,    # lai2
            '去': 4,    # qu4
            '说': 1,    # shuo1
            '看': 4,    # kan4
            '听': 1,    # ting1
            '吃': 1,    # chi1
            '喝': 1,    # he1
            '买': 3,    # mai3
            '卖': 4,    # mai4
            '学': 2,    # xue2
            '习': 2,    # xi2
            '工': 1,    # gong1
            '作': 4,    # zuo4
            '家': 1,    # jia1
            '回': 2,    # hui2
            '走': 3,    # zou3
            '跑': 3,    # pao3
            '飞': 1,    # fei1
            '开': 1,    # kai1
            '关': 1,    # guan1
            '门': 2,    # men2
            '窗': 1,    # chuang1
            '书': 1,    # shu1
            '桌': 1,    # zhuo1
            '椅': 3,    # yi3
            '床': 2,    # chuang2
            '房': 2,    # fang2
            '间': 1,    # jian1
            '水': 3,    # shui3
            '火': 3,    # huo3
            '电': 4,    # dian4
            '话': 4,    # hua4
            '车': 1,    # che1
            '船': 2,    # chuan2
            '飞机': 1,  # feiji1 (but this is two characters)
            '手': 3,    # shou3
            '脚': 3,    # jiao3
            '头': 2,    # tou2
            '眼': 3,    # yan3
            '耳': 3,    # er3
            '口': 3,    # kou3
            '鼻': 4,    # bi4
        }
        
        return tone_dict
    
    def get_character_with_timing(self, audio_path):
        """Get Chinese characters with timing from wav2vec2"""
        try:
            # Load audio for wav2vec2
            audio = librosa.load(audio_path, sr=16000)[0]
            audio = librosa.util.normalize(audio)
            
            # Get predictions
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            print(f"Wav2vec2 transcription: '{transcription}'")
            
            # Get character timings
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
                
                # Skip padding and special tokens
                if token in ['<pad>', '<s>', '</s>', '<unk>', '|', '[PAD]']:
                    continue
                
                if token != current_char and token.strip():  # Non-empty token
                    # End previous character
                    if current_char is not None and current_char.strip():
                        char_timings.append({
                            'char': current_char,
                            'start_time': char_start * frame_duration,
                            'end_time': i * frame_duration,
                            'duration': (i - char_start) * frame_duration
                        })
                    
                    # Start new character
                    current_char = token
                    char_start = i
        
        # Handle last character
        if current_char is not None and current_char.strip():
            char_timings.append({
                'char': current_char,
                'start_time': char_start * frame_duration,
                'end_time': len(frame_predictions) * frame_duration,
                'duration': (len(frame_predictions) - char_start) * frame_duration
            })
        
        return char_timings
    
    def lookup_character_tones(self, char_timings):
        """Look up tones for detected characters"""
        results = []
        
        for i, char_timing in enumerate(char_timings):
            char = char_timing['char']
            
            # Look up tone in dictionary
            if char in self.tone_dict:
                tone = self.tone_dict[char]
                confidence = 1.0  # Dictionary lookup is definitive
                method = "dictionary_lookup"
            else:
                # Unknown character - could implement fallback logic here
                tone = None
                confidence = 0.0
                method = "unknown"
            
            result = {
                'syllable_idx': i,
                'character': char,
                'pinyin': self.char_to_pinyin(char),
                'predicted_tone': tone,
                'confidence': confidence,
                'start_time': char_timing['start_time'],
                'end_time': char_timing['end_time'],
                'duration': char_timing['duration'],
                'method': method
            }
            
            results.append(result)
        
        return results
    
    def char_to_pinyin(self, char):
        """Convert character to pinyin (basic mapping)"""
        # This is a simplified mapping - in practice you'd use a comprehensive pinyin library
        char_to_pinyin_dict = {
            '我': 'wo',
            '矮': 'ai', 
            '爱': 'ai',
            '猫': 'mao',
            '毛': 'mao',
            '帽': 'mao',
            '你': 'ni',
            '好': 'hao',
            '很': 'hen',
            '大': 'da',
            '小': 'xiao',
            # Add more mappings as needed
        }
        
        return char_to_pinyin_dict.get(char, char)  # Return character if no pinyin found
    
    def predict_tones_from_characters(self, audio_path):
        """Main function: detect characters and look up their tones"""
        print(f"Analyzing: {os.path.basename(audio_path)}")
        
        # Get characters with timing
        transcription, char_timings = self.get_character_with_timing(audio_path)
        
        if not char_timings:
            print("No characters detected")
            return None
        
        print(f"Detected characters: {[ct['char'] for ct in char_timings]}")
        
        # Look up tones
        results = self.lookup_character_tones(char_timings)
        
        return results
    
    def format_results(self, results):
        """Format results for display"""
        if not results:
            return "No results available"
        
        output = []
        output.append("Chinese Character Tone Analysis:")
        output.append("=" * 50)
        output.append(f"Total characters: {len(results)}")
        output.append("")
        
        for result in results:
            char = result['character']
            pinyin = result['pinyin']
            tone = result['predicted_tone']
            confidence = result['confidence']
            method = result['method']
            
            if tone is not None:
                output.append(f"Character: {char}")
                output.append(f"  Pinyin: {pinyin}")
                output.append(f"  Tone: T{tone}")
                output.append(f"  Confidence: {confidence:.3f}")
                output.append(f"  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
                output.append(f"  Method: {method}")
            else:
                output.append(f"Character: {char}")
                output.append(f"  Pinyin: {pinyin}")
                output.append(f"  Tone: UNKNOWN")
                output.append(f"  Method: {method}")
            output.append("")
        
        # Summary
        known_results = [r for r in results if r['predicted_tone'] is not None]
        if known_results:
            chars = [r['character'] for r in known_results]
            pinyins = [r['pinyin'] for r in known_results]
            tones = [str(r['predicted_tone']) for r in known_results]
            
            output.append("SUMMARY:")
            output.append(f"Characters: {''.join(chars)}")
            output.append(f"Pinyin: {' '.join(pinyins)}")
            output.append(f"Tones: {' '.join(tones)}")
            output.append(f"Pinyin with tones: {' '.join([f'{p}{t}' for p, t in zip(pinyins, tones)])}")
            output.append(f"Known characters: {len(known_results)}/{len(results)}")
        
        return "\n".join(output)
    
    def add_character_tone(self, char, tone):
        """Add new character-tone mapping to dictionary"""
        self.tone_dict[char] = tone
        print(f"Added: {char} -> T{tone}")
    
    def save_tone_dictionary(self, filename="custom_tone_dict.json"):
        """Save current tone dictionary to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.tone_dict, f, ensure_ascii=False, indent=2)
        print(f"Tone dictionary saved to {filename}")
    
    def load_custom_dictionary(self, filename="custom_tone_dict.json"):
        """Load custom tone dictionary"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                custom_dict = json.load(f)
            self.tone_dict.update(custom_dict)
            print(f"Loaded custom dictionary from {filename}")
        except FileNotFoundError:
            print(f"Custom dictionary {filename} not found")

def main():
    """Test character-based tone detection"""
    
    print("=" * 60)
    print("CHINESE CHARACTER TONE LOOKUP SYSTEM")
    print("=" * 60)
    
    # Initialize
    classifier = ChineseCharToneLookup()
    
    # Test file
    audio_path = r"C:\Users\maruk\carfac-SAI\python\src\carfac\reference\woaimao.mp3"
    
    if os.path.exists(audio_path):
        results = classifier.predict_tones_from_characters(audio_path)
        
        if results:
            print("\n" + classifier.format_results(results))
            
            # Interactive mode to add unknown characters
            unknown_chars = [r for r in results if r['predicted_tone'] is None]
            if unknown_chars:
                print(f"\nFound {len(unknown_chars)} unknown characters:")
                for result in unknown_chars:
                    char = result['character']
                    pinyin = result['pinyin']
                    print(f"Character: {char} (pinyin: {pinyin})")
                    
                    try:
                        tone = input(f"Enter tone for '{char}' (1-5, or 'skip'): ").strip()
                        if tone.isdigit() and 1 <= int(tone) <= 5:
                            classifier.add_character_tone(char, int(tone))
                        elif tone.lower() != 'skip':
                            print("Invalid tone. Skipped.")
                    except (ValueError, KeyboardInterrupt):
                        print("Skipped.")
                
                # Save updated dictionary
                save = input("\nSave updated dictionary? (y/n): ").strip().lower()
                if save == 'y':
                    classifier.save_tone_dictionary()
        else:
            print("No characters detected")
    else:
        print(f"Audio file not found: {audio_path}")

if __name__ == "__main__":
    main()