import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
import glob

class CorrectedPaperMethod:
    """Corrected implementation of the paper's exact methodology"""
    
    def __init__(self):
        self.syllable_to_idx = {}
        self.model = None
        
    def extract_mel_spectrogram_corrected(self, audio_path):
        """Extract Mel-spectrogram with corrected parameters to avoid warnings"""
        try:
            # Load audio - use higher sampling rate to avoid filter warnings
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            
            if len(y) < 0.5 * sr:
                return None, sr
            
            # Trim and normalize
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)
            
            # Paper's mel-spectrogram settings adapted for higher sr
            # Scale frequency range proportionally: [50,350] -> [72, 504] for 22050 Hz
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=64,
                fmin=72,    # Scaled from 50
                fmax=504,   # Scaled from 350
                hop_length=int(sr * 0.013),  # 13ms hop
                n_fft=int(sr * 0.025),       # 25ms window
                window='hann'
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return log_mel_spec.T, sr  # (time, 64)
            
        except Exception as e:
            print(f"Mel-spectrogram extraction error: {e}")
            return None, 22050
    
    def create_tri_tone_segments_corrected(self, mel_spec, syllable_info):
        """Create tri-tone segments with proper error handling"""
        if mel_spec is None or len(mel_spec) == 0:
            return []
        
        # For single syllable (most of your data), create artificial boundaries
        total_frames = mel_spec.shape[0]
        
        # Divide into 3 parts for tri-tone context
        segment_length = total_frames // 3
        
        segments = []
        
        # Create 3 segments: previous half + current + next half
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
    
    def create_context_features_corrected(self, segments, context_size=1):
        """Create contextual features with proper padding"""
        if not segments:
            return []
        
        contextual_data = []
        context_length = context_size * 2 + 1  # 1 context = 3 total
        
        for center_idx in range(len(segments)):
            context_segments = []
            context_durations = []
            context_syllable_ids = []
            
            # Collect context window
            for offset in range(-context_size, context_size + 1):
                idx = center_idx + offset
                
                if 0 <= idx < len(segments):
                    # Use actual segment
                    context_segments.append(segments[idx]['segment'])
                    context_durations.append(segments[idx]['duration'])
                    context_syllable_ids.append(segments[idx]['syllable_id'])
                else:
                    # Use padding - duplicate edge segments
                    if idx < 0:
                        # Use first segment for padding
                        context_segments.append(segments[0]['segment'])
                        context_durations.append(segments[0]['duration'])
                        context_syllable_ids.append(segments[0]['syllable_id'])
                    else:
                        # Use last segment for padding
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
    
    def build_resnet_sp_corrected(self, input_shape):
        """Build ResNetSP with corrected architecture"""
        inputs = layers.Input(shape=input_shape, name='mel_input')
        
        # Initial processing
        x = layers.Conv1D(64, 7, strides=2, padding='same', name='initial_conv')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.Activation('relu', name='initial_relu')(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same', name='initial_pool')(x)
        
        # ResNet blocks
        filter_sizes = [64, 128, 256]
        
        for stage, filters in enumerate(filter_sizes):
            # Residual block 1
            residual = x
            x = layers.Conv1D(filters, 3, padding='same', name=f'stage{stage}_conv1a')(x)
            x = layers.BatchNormalization(name=f'stage{stage}_bn1a')(x)
            x = layers.Activation('relu', name=f'stage{stage}_relu1a')(x)
            x = layers.Conv1D(filters, 3, padding='same', name=f'stage{stage}_conv1b')(x)
            x = layers.BatchNormalization(name=f'stage{stage}_bn1b')(x)
            
            # Skip connection with proper dimension matching
            if residual.shape[-1] != filters:
                residual = layers.Conv1D(filters, 1, name=f'stage{stage}_skip1')(residual)
            x = layers.Add(name=f'stage{stage}_add1')([x, residual])
            x = layers.Activation('relu', name=f'stage{stage}_relu1c')(x)
            
            # Residual block 2
            residual = x
            x = layers.Conv1D(filters, 3, padding='same', name=f'stage{stage}_conv2a')(x)
            x = layers.BatchNormalization(name=f'stage{stage}_bn2a')(x)
            x = layers.Activation('relu', name=f'stage{stage}_relu2a')(x)
            x = layers.Conv1D(filters, 3, padding='same', name=f'stage{stage}_conv2b')(x)
            x = layers.BatchNormalization(name=f'stage{stage}_bn2b')(x)
            x = layers.Add(name=f'stage{stage}_add2')([x, residual])
            x = layers.Activation('relu', name=f'stage{stage}_relu2c')(x)
            
            # Downsampling
            x = layers.MaxPooling1D(2, name=f'stage{stage}_pool')(x)
        
        # Global pooling and embedding
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        x = layers.Dense(128, activation='relu', name='embedding_dense')(x)
        x = layers.Dropout(0.3, name='embedding_dropout')(x)
        
        return models.Model(inputs, x, name='ResNetSP')
    
    def build_contextual_model_corrected(self, num_syllables, context_length):
        """Build contextual model with proper input handling"""
        # Duration features
        duration_input = layers.Input(shape=(context_length,), name='durations')
        
        # Syllable ID features  
        syllable_input = layers.Input(shape=(context_length,), name='syllables')
        
        # Embedding for syllables
        syllable_embeddings = layers.Embedding(
            input_dim=num_syllables,
            output_dim=32,
            name='syllable_embedding'
        )(syllable_input)
        
        # Process durations
        duration_expanded = layers.Reshape((context_length, 1))(duration_input)
        
        # Combine features
        combined = layers.Concatenate(axis=-1)([duration_expanded, syllable_embeddings])
        combined_flat = layers.Flatten()(combined)
        
        # MLP layers
        x = layers.Dense(256, activation='relu', name='context_dense1')(combined_flat)
        x = layers.BatchNormalization(name='context_bn1')(x)
        x = layers.Dropout(0.4, name='context_dropout1')(x)
        
        x = layers.Dense(128, activation='relu', name='context_dense2')(x)
        x = layers.Dropout(0.3, name='context_dropout2')(x)
        
        return models.Model([duration_input, syllable_input], x, name='ContextualModel')
    
    def build_complete_model_corrected(self, mel_input_shape, num_syllables, context_size=1, num_classes=4):
        """Build complete model with proper architecture"""
        context_length = context_size * 2 + 1
        
        # Sub-models
        resnet_sp = self.build_resnet_sp_corrected(mel_input_shape)
        contextual_model = self.build_contextual_model_corrected(num_syllables, context_length)
        
        # Main inputs
        mel_inputs = []
        for i in range(context_length):
            mel_input = layers.Input(shape=mel_input_shape, name=f'mel_context_{i}')
            mel_inputs.append(mel_input)
        
        duration_input = layers.Input(shape=(context_length,), name='durations')
        syllable_input = layers.Input(shape=(context_length,), name='syllables')
        
        # Process mel-spectrograms
        mel_embeddings = []
        for i, mel_input in enumerate(mel_inputs):
            embedding = resnet_sp(mel_input)
            mel_embeddings.append(embedding)
        
        # Combine mel embeddings
        if len(mel_embeddings) > 1:
            combined_mel = layers.Concatenate(name='mel_concat')(mel_embeddings)
        else:
            combined_mel = mel_embeddings[0]
        
        # Get context features
        context_features = contextual_model([duration_input, syllable_input])
        
        # Final combination
        final_features = layers.Concatenate(name='final_concat')([combined_mel, context_features])
        
        # Final classifier (paper mentions 128 units)
        x = layers.Dense(128, activation='relu', name='final_dense')(final_features)
        x = layers.BatchNormalization(name='final_bn')(x)
        x = layers.Dropout(0.4, name='final_dropout')(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        all_inputs = mel_inputs + [duration_input, syllable_input]
        model = models.Model(inputs=all_inputs, outputs=outputs, name='PaperToneClassifier')
        
        # Compile with paper's settings
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_dataset_corrected(self, data_dir, context_size=1, max_samples=1000):
        """Prepare dataset with corrected processing"""
        print("Dataset preparation (Corrected Paper Method)...")
        
        # Collect files
        audio_files = []
        for ext in ['*.mp3', '*.wav']:
            audio_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Build vocabulary
        syllables = set()
        for file_path in audio_files:
            filename = os.path.basename(file_path)
            try:
                word_tone = filename.split('_')[0]
                syllable = word_tone[:-1] if word_tone[-1].isdigit() else word_tone
                syllables.add(syllable)
            except:
                continue
        
        syllables.add('<PAD>')
        self.syllable_to_idx = {syl: idx for idx, syl in enumerate(sorted(syllables))}
        print(f"Syllable vocabulary size: {len(self.syllable_to_idx)}")
        
        # Process data
        X_mel_contexts = []
        X_durations = []
        X_syllable_ids = []
        y = []
        
        context_length = context_size * 2 + 1
        
        for file_path in tqdm(audio_files[:max_samples], desc="Processing"):
            try:
                filename = os.path.basename(file_path)
                word_tone = filename.split('_')[0]
                
                if not word_tone[-1].isdigit():
                    continue
                
                tone = int(word_tone[-1])
                if tone < 1 or tone > 4:
                    continue
                
                tone_label = tone - 1  # 0-indexed
                syllable = word_tone[:-1]
                
                # Extract mel-spectrogram
                mel_spec, sr = self.extract_mel_spectrogram_corrected(file_path)
                if mel_spec is None:
                    continue
                
                # Create syllable info
                syllable_info = {
                    'syllable': syllable,
                    'syllable_id': self.syllable_to_idx.get(syllable, 0)
                }
                
                # Create tri-tone segments
                segments = self.create_tri_tone_segments_corrected(mel_spec, syllable_info)
                if not segments:
                    continue
                
                # Create context features
                contextual_data = self.create_context_features_corrected(segments, context_size)
                
                for ctx_data in contextual_data:
                    # Standardize segment lengths
                    max_length = 150  # Reduced for stability
                    
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
                    
                    # Prepare other features
                    durations = ctx_data['context_durations'][:]
                    while len(durations) < context_length:
                        durations.append(0.0)
                    durations = durations[:context_length]
                    
                    syllable_ids = ctx_data['context_syllable_ids'][:]
                    while len(syllable_ids) < context_length:
                        syllable_ids.append(0)
                    syllable_ids = syllable_ids[:context_length]
                    
                    X_mel_contexts.append(mel_contexts)
                    X_durations.append(durations)
                    X_syllable_ids.append(syllable_ids)
                    y.append(tone_label)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Processing complete: {len(X_mel_contexts)} samples")
        
        if len(X_mel_contexts) == 0:
            return None
        
        # Convert to arrays
        X_mel_contexts = np.array(X_mel_contexts, dtype=np.float32)
        X_durations = np.array(X_durations, dtype=np.float32)
        X_syllable_ids = np.array(X_syllable_ids, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        
        print(f"Final shapes:")
        print(f"  Mel contexts: {X_mel_contexts.shape}")
        print(f"  Durations: {X_durations.shape}")
        print(f"  Syllable IDs: {X_syllable_ids.shape}")
        print(f"  Labels: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X_mel_contexts, X_durations, X_syllable_ids, y
    
    def train_corrected_paper_method(self, data_dir, context_size=1, max_samples=1000):
        """Train using corrected paper method"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"corrected_paper_models/paper_corrected_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("CORRECTED Paper Method Implementation")
        print("=" * 60)
        
        # Prepare dataset
        dataset = self.prepare_dataset_corrected(data_dir, context_size, max_samples)
        if dataset is None:
            return None, None
        
        X_mel_contexts, X_durations, X_syllable_ids, y = dataset
        
        # Data splitting
        X_train_mel, X_temp_mel, X_train_dur, X_temp_dur, X_train_syl, X_temp_syl, y_train, y_temp = train_test_split(
            X_mel_contexts, X_durations, X_syllable_ids, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val_mel, X_test_mel, X_val_dur, X_test_dur, X_val_syl, X_test_syl, y_val, y_test = train_test_split(
            X_temp_mel, X_temp_dur, X_temp_syl, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Data split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
        
        # Build model
        mel_input_shape = X_mel_contexts.shape[2:]
        num_syllables = len(self.syllable_to_idx)
        num_classes = len(np.unique(y))
        
        print(f"Model specs: Mel shape={mel_input_shape}, Syllables={num_syllables}, Classes={num_classes}")
        
        self.model = self.build_complete_model_corrected(
            mel_input_shape, num_syllables, context_size, num_classes
        )
        
        # Prepare inputs
        context_length = context_size * 2 + 1
        
        train_inputs = []
        val_inputs = []
        test_inputs = []
        
        for i in range(context_length):
            train_inputs.append(X_train_mel[:, i])
            val_inputs.append(X_val_mel[:, i])
            test_inputs.append(X_test_mel[:, i])
        
        train_inputs.extend([X_train_dur, X_train_syl])
        val_inputs.extend([X_val_dur, X_val_syl])
        test_inputs.extend([X_test_dur, X_test_syl])
        
        # Class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8),
            ModelCheckpoint(f'{save_dir}/best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Training
        print("\nStarting training...")
        history = self.model.fit(
            train_inputs, y_train,
            validation_data=(val_inputs, y_val),
            batch_size=16,  # Smaller batch for stability
            epochs=100,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluation
        test_loss, test_acc = self.model.evaluate(test_inputs, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        y_pred = self.model.predict(test_inputs)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_classes))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=['T1', 'T2', 'T3', 'T4']))
        
        # Save
        self.model.save(f'{save_dir}/corrected_paper_model.keras')
        
        config = {
            'syllable_to_idx': self.syllable_to_idx,
            'context_size': context_size,
            'mel_input_shape': mel_input_shape,
            'num_classes': num_classes
        }
        
        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nModel saved: {save_dir}")
        return self.model, save_dir

def main():
    data_dir = r'C:\Users\maruk\Downloads\tone_perfect_all_mp3\tone_perfect'
    
    print("=== CORRECTED Paper Method ===")
    choice = input("Context size (1=fast, 2=medium, 3=slow): ").strip() or "1"
    max_samples = int(input("Max samples (500/1000/2000): ").strip() or "1000")
    
    context_size = int(choice)
    
    classifier = CorrectedPaperMethod()
    model, save_dir = classifier.train_corrected_paper_method(
        data_dir, context_size=context_size, max_samples=max_samples
    )
    
    if model:
        print("\nCorrected paper method training completed!")
    else:
        print("\nTraining failed")

if __name__ == "__main__":
    main()