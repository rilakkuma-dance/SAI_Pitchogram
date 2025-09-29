import os
import numpy as np
import librosa
import tensorflow as tf
import json
import glob
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# model dir is in the same dir as this file
# so we can use __file__ to get current path
# and then load model and config from there
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(MODULES_DIR, "tone_detection_model", "tone_detection_model_20250928_113915")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
MODEL_FILE = os.path.join(MODEL_DIR, "tone_detection.keras")

class ToneClassifierTester:
    def __init__(self, model_dir=MODEL_FILE, config_path=CONFIG_PATH):
        """Load the trained model and configuration"""
        self.model = tf.keras.models.load_model(model_dir)
        
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
    
    def extract_mel_spectrogram(self, audio_input):
        """Extract mel-spectrogram using the same method as training
        
        Args:
            audio_input: str (file path), np.ndarray, or tensor
        """
        try:
            # Process different input types
            if isinstance(audio_input, str):
                # File path - load using librosa
                y, sr = librosa.load(audio_input, sr=22050, duration=3.0)
            else:
                # Handle numpy array or tensor
                if hasattr(audio_input, 'cpu'):  # torch tensor
                    y = audio_input.cpu().numpy()
                else:  # numpy array
                    y = np.array(audio_input)
                
                if y.ndim > 1:
                    y = y[0]  # Take first channel
                
                sr = 22050  # Assume 22050 Hz and resample if needed
                y = librosa.resample(y, orig_sr=len(y)//3 if len(y) > 66150 else 22050, target_sr=22050)
                
                # Limit to 3 seconds
                if len(y) > 22050 * 3:
                    y = y[:22050 * 3]
            
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
    
    def predict_tone(self, audio_input, syllable_text="unknown"):
        """Predict tone for audio input
        
        Args:
            audio_input: str (file path), np.ndarray, or tensor
            syllable_text: str
        """
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio_input)
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
    """Main testing function for a single audio file"""
    
    # Base directory = project root (where this script lives, two levels up)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Model + config relative to BASE_DIR
    model_dir = os.path.join(BASE_DIR, "tone_detection_model", "tone_detection_model_20250928_113915", "tone_detection.keras")
    config_path = os.path.join(BASE_DIR, "tone_detection_model", "tone_detection_model_20250928_113915", "config.json")

    # Check if model files exist
    if not os.path.exists(model_dir):
        print(f"Model file not found: {model_dir}")
        return
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    print("=" * 60)
    print("Tone Classifier - Single File Test")
    print("=" * 60)
    
    # Load model
    tester = ToneClassifierTester(model_dir, config_path)
    
    # Ask for one file to test
    file_path = input("\nEnter audio file path: ").strip()
    if os.path.exists(file_path):
        result = tester.predict_tone(file_path)
        if result:
            print(f"\nPrediction: T{result['predicted_tone']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("All probabilities:")
            for tone, prob in result['class_probabilities'].items():
                print(f"  {tone}: {prob:.3f}")
        else:
            print("Failed to process file")
    else:
        print("File not found")

if __name__ == "__main__":
    main()