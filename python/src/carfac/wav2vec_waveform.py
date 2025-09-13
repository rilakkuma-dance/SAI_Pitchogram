import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import threading
import queue
import time
from dataclasses import dataclass
import librosa
import argparse
import os
import sounddevice as sd
import wave
from datetime import datetime
import speech_recognition as sr
from collections import deque

# Font setup
def get_font_path():
    """Get font path relative to script location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "fonts", "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "DoulosSIL-7.000", "DoulosSIL-Regular.ttf"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Initialize font
font_path = get_font_path()

if font_path:
    font_prop = fm.FontProperties(fname=font_path, size=16)
    print(f"Using Doulos SIL font: {font_path}")
else:
    font_prop = fm.FontProperties(family='Times New Roman', size=16)
    print("Doulos SIL not found, using Times New Roman fallback")

# Configure matplotlib
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman', 'Arial Unicode MS', 'Segoe UI', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 12
})

# Wav2Vec2 imports
try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
    WAV2VEC2_AVAILABLE = True
except ImportError:
    print("Warning: torch/transformers not found. Install with: pip install torch torchaudio transformers")
    WAV2VEC2_AVAILABLE = False

# JAX/CARFAC imports
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    import carfac.jax.carfac as carfac
    from carfac.np.carfac import CarParams
    import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Using simplified visualization.")
    JAX_AVAILABLE = False

@dataclass
class SAIParams:
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

# ---------------- Simple Audio Processor for Fallback ----------------
class SimpleAudioProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        self.n_channels = 200
        self.frequencies = np.logspace(np.log10(50), np.log10(8000), self.n_channels)

    def process_chunk(self, audio_chunk):
        if len(audio_chunk) == 0:
            return np.zeros((self.n_channels, 1))
        
        # Simple frequency analysis using FFT
        fft = np.fft.fft(audio_chunk)
        freqs = np.fft.fftfreq(len(audio_chunk), 1/self.fs)
        magnitude = np.abs(fft)
        
        # Map to cochlear channels
        output = np.zeros((self.n_channels, 1))
        for i, freq in enumerate(self.frequencies):
            # Find closest frequency bins
            freq_mask = (freqs >= freq * 0.8) & (freqs <= freq * 1.2)
            if np.any(freq_mask):
                output[i, 0] = np.mean(magnitude[freq_mask])
        
        # Normalize
        if np.max(output) > 0:
            output = output / np.max(output)
        
        return output

# ---------------- Chunk-based Wav2Vec2 Phoneme Handler ----------------
class SimpleWav2Vec2Handler:
    """Wav2Vec2 phoneme recognition handler - records all audio then processes once"""
    
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.model_name = model_name
        self.enabled = WAV2VEC2_AVAILABLE
        
        if not self.enabled:
            print("Wav2Vec2 not available")
            return
        
        # Simple state
        self.is_recording = False
        self.is_processing = False
        self.result = None
        
        # Audio setup
        self.sample_rate = 16000
        self.audio_data = []
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=16000)
        
        # Wav2Vec2 model components
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
        # Calibrate microphone
        try:
            with self.microphone:
                self.recognizer.adjust_for_ambient_noise(self.microphone)
            print("Microphone ready")
        except Exception as e:
            print(f"Microphone warning: {e}")
        
        # Pre-load Wav2Vec2 model during initialization
        print("Initializing Wav2Vec2 model...")
        if not self.load_model():
            print("Warning: Failed to load model during initialization")

    def load_model(self):
        """Load Wav2Vec2 model"""
        if not self.enabled:
            print("Wav2Vec2 not enabled, skipping model load")
            return False
        
        if self.model is None:
            try:
                print(f"Loading Wav2Vec2 {self.model_name}...")
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                print("Model loaded successfully")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.enabled = False
                return False
        else:
            print("Model already loaded")
            return True

    def get_current_status(self):
        """Simple 3-state status"""
        if not self.enabled:
            return "Wav2Vec2 not available", "red"
        
        if self.is_recording:
            return "Recording...", "yellow"
        
        if self.is_processing:
            return "Processing phonemes...", "orange"
        
        if self.result is not None:
            if self.result == "no_audio":
                return "No audio detected", "gray"
            else:
                return f"Phonemes: {self.result}", "green"
        
        return "Ready to start", "blue"

    def start_recording(self):
        """Start recording"""
        if not self.enabled or self.is_recording:
            return False
        
        # Model should already be loaded during init
        if self.model is None:
            print("Error: Model not loaded")
            return False
        
        # Reset state
        self.is_recording = True
        self.is_processing = False
        self.result = None
        self.audio_data = []
        
        # Start recording thread
        threading.Thread(target=self._record_audio, daemon=True).start()
        return True

    def stop_recording(self):
        """Stop recording and process"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.is_processing = True
        
        # Start processing thread
        threading.Thread(target=self._process_audio, daemon=True).start()

    def _record_audio(self):
        """Record audio continuously"""
        try:
            with self.microphone as source:
                stream = source.stream
                while self.is_recording:
                    # Read a small chunk of audio continuously
                    audio_chunk = stream.read(source.CHUNK)
                    audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_data.append(audio_np)

        except Exception as e:
            print(f"Recording error: {e}")

    def _process_audio(self):
        """Process all recorded audio with Wav2Vec2"""
        try:
            if not self.audio_data:
                self.result = "no_audio"
                self.is_processing = False
                return
            
            # Combine all audio
            complete_audio = np.concatenate(self.audio_data)
            
            # Save audio file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"recording_{timestamp}.wav"
            self._save_audio(complete_audio, filename)
            
            # Convert to tensor for Wav2Vec2
            waveform = torch.tensor(complete_audio).float()
            
            # Prepare input
            input_values = self.feature_extractor(
                waveform, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_values
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode phonemes
            predicted_ids = torch.argmax(logits, dim=-1)
            phonemes = self.tokenizer.batch_decode(predicted_ids)[0]
            
            if phonemes and len(phonemes.strip()) > 0:
                self.result = phonemes.strip()
                print(f"Phonemes detected: {self.result}")
            else:
                self.result = "no_audio"
                print("No phonemes detected")
                
        except Exception as e:
            print(f"Processing error: {e}")
            self.result = "no_audio"
        
        self.is_processing = False

    def _save_audio(self, audio_data, filename):
        """Save audio to file"""
        try:
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Convert to 16-bit
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Save
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"Audio saved: {filename}")
        except Exception as e:
            print(f"Save error: {e}")

# ---------------- Audio Processor with Fallback ----------------
class AudioProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        if JAX_AVAILABLE:
            try:
                self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
                    carfac.CarfacDesignParameters(fs=fs, n_ears=1)
                )
                self.n_channels = self.hypers.ears[0].car.n_ch
                self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])
                self.use_carfac = True
                print("Using CARFAC audio processing")
            except Exception as e:
                print(f"CARFAC initialization failed: {e}")
                self.use_carfac = False
        else:
            self.use_carfac = False
        
        if not self.use_carfac:
            self.fallback = SimpleAudioProcessor(fs=fs)
            self.n_channels = self.fallback.n_channels
            print("Using simple audio processing")

    def process_chunk(self, audio_chunk):
        if self.use_carfac:
            try:
                if len(audio_chunk.shape) == 1:
                    audio_input = audio_chunk.reshape(-1, 1)
                else:
                    audio_input = audio_chunk
                audio_jax = jnp.array(audio_input, dtype=jnp.float32)
                naps, _, self.state, _, _, _ = self.run_segment_jit(
                    audio_jax, self.hypers, self.weights, self.state, open_loop=False
                )
                return np.array(naps[:, :, 0]).T
            except Exception as e:
                print(f"CARFAC processing error: {e}")
                return self.fallback.process_chunk(audio_chunk)
        else:
            return self.fallback.process_chunk(audio_chunk)

# ---------------- SAI Processor with Fallback ----------------
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
                print("Using SAI processing")
            except Exception as e:
                print(f"SAI initialization failed: {e}")
                self.use_sai = False
        else:
            self.use_sai = False
        
        if not self.use_sai:
            print("Using simple autocorrelation")
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception as e:
                print(f"SAI processing error: {e}")
                return self._simple_sai(nap_output)
        else:
            return self._simple_sai(nap_output)
    
    def _simple_sai(self, nap_output):
        sai_output = np.zeros((self.sai_params.num_channels, self.sai_params.sai_width))
        
        for ch in range(min(nap_output.shape[0], self.sai_params.num_channels)):
            if nap_output.shape[1] > 0:
                channel_data = nap_output[ch, :]
                for lag in range(min(len(channel_data), self.sai_params.sai_width)):
                    if len(channel_data) > lag:
                        start_idx = max(0, len(channel_data) - lag - 10)
                        end_idx = len(channel_data) - lag
                        if end_idx > start_idx:
                            sai_output[ch, lag] = np.mean(channel_data[start_idx:end_idx])
        
        return sai_output

# ---------------- Visualization Handler ----------------
class VisualizationHandler:
    def __init__(self, sample_rate_hz: int, sai_params: SAIParams):
        self.sample_rate_hz = sample_rate_hz
        self.sai_params = sai_params
        self.output = np.zeros(200)
        self.img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.vowel_coords = np.zeros((2, 1), dtype=np.float32)

    def get_vowel_embedding(self, nap):
        if nap.shape[0] > 0:
            self.vowel_coords[0, 0] = np.mean(nap[:10, :]) if nap.shape[1] > 0 else 0
            self.vowel_coords[1, 0] = np.mean(nap[-10:, :]) if nap.shape[1] > 0 else 0
        return self.vowel_coords

    def run_frame(self, sai_frame: np.ndarray):
        if sai_frame.size > 0:
            self.output = sai_frame.mean(axis=0)[:len(self.output)]
        return self.output

    def draw_column(self, column_ptr: np.ndarray):
        v = np.ravel(self.vowel_coords)
        tint = np.array([
            0.5 - 0.6 * (v[1] if len(v) > 1 else 0),
            0.5 - 0.6 * (v[0] if len(v) > 0 else 0),
            0.35 * (v[0] + v[1] if len(v) > 1 else 0) + 0.4
        ], dtype=np.float32)
        k_scale = 0.5 * 255
        tint *= k_scale
        
        for i in range(min(len(self.output), len(column_ptr))):
            column_ptr[i] = np.clip(np.int32((tint * self.output[i])), 0, 255)

# ---------------- Waveform Buffer Class ----------------
class WaveformBuffer:
    """Circular buffer for storing waveform data for display"""
    def __init__(self, size=8000):  # ~0.5 seconds at 16kHz
        self.size = size
        self.buffer = np.zeros(size)
        self.index = 0
        
    def add_chunk(self, chunk):
        """Add a chunk of audio data to the circular buffer"""
        chunk_size = len(chunk)
        if chunk_size >= self.size:
            # If chunk is larger than buffer, just take the last part
            self.buffer = chunk[-self.size:].copy()
            self.index = 0
        else:
            # Add chunk to buffer
            end_idx = self.index + chunk_size
            if end_idx <= self.size:
                # Chunk fits without wrapping
                self.buffer[self.index:end_idx] = chunk
                self.index = end_idx % self.size
            else:
                # Chunk needs to wrap around
                first_part = self.size - self.index
                self.buffer[self.index:] = chunk[:first_part]
                self.buffer[:chunk_size - first_part] = chunk[first_part:]
                self.index = chunk_size - first_part
    
    def get_waveform(self):
        """Get the current waveform data in correct order"""
        if self.index == 0:
            return self.buffer.copy()
        else:
            return np.concatenate([self.buffer[self.index:], self.buffer[:self.index]])

# ---------------- Main SAI Visualization with Wav2Vec2 ----------------
class SAIVisualizationWithWav2Vec2:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 debug=True, playback_speed=3.0, loop_audio=True, wav2vec2_model="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        self.sai_speed = 1.0  # Independent SAI visualization speed
        self.sai_file_index = 0.0  # SAI frame accumulator

        # Reference text and target phonemes for similarity
        self.reference_text = None
        self.reference_pronunciation = None
        self.translated_text = None
        self.target_phonemes = "ɕiɛ5ɕiɛ5"  # Target for xiè xiè
        
        # Initialize Wav2Vec2 handler
        self.wav2vec2_handler = SimpleWav2Vec2Handler(model_name=wav2vec2_model)
        
        # Audio processors
        self.processor_realtime = AudioProcessor(fs=sample_rate)
        self.processor_file = AudioProcessor(fs=sample_rate)
        self.n_channels = self.processor_realtime.n_channels

        # SAI parameters
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=self.sai_width,
            future_lags=self.sai_width - 1,
            num_triggers_per_frame=2,
            trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size,
            channel_smoothing_scale=0.5
        )
        
        # SAI processors
        self.sai_realtime = SAIProcessor(self.sai_params)
        self.sai_file = SAIProcessor(self.sai_params)

        # Visualization
        self.vis_realtime = VisualizationHandler(sample_rate, self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, self.sai_params)

        # Waveform buffers for display
        self.waveform_realtime = WaveformBuffer(size=int(sample_rate * 0.5))  # 0.5 seconds
        self.waveform_file = WaveformBuffer(size=int(sample_rate * 0.5))

        # Audio setup for SAI visualization only
        self.audio_queue = queue.Queue(maxsize=50)
        
        # File processing
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.current_position = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0
        
        # Audio playback
        self.audio_playback_enabled = True
        self.audio_output_stream = None
        self.playback_position = 0.0
        
        if audio_file_path and os.path.exists(audio_file_path):
            self._load_audio_file()
        
        # PyAudio for SAI visualization
        self.p = None
        self.stream = None
        self.running = False
        
        # Initialize similarity display as None
        self.similarity_display = None
        self.similarity_rect = None
        
        self._setup_dual_visualization()

    def decrease_sai_speed(self, event=None):
        """Decrease SAI visualization speed"""
        self.sai_speed = max(0.1, self.sai_speed - 0.25)
        self.update_sai_speed_display()
        print(f"SAI speed: {self.sai_speed:.1f}x")

    def increase_sai_speed(self, event=None):
        """Increase SAI visualization speed"""
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
        self.update_sai_speed_display()
        print(f"SAI speed: {self.sai_speed:.1f}x")

    def update_sai_speed_display(self):
        """Update SAI speed display"""
        if hasattr(self, 'sai_speed_display'):
            self.sai_speed_display.set_text(f'SAI Speed: {self.sai_speed:.1f}x')

    # Audio speed control methods (renamed from existing)
    def decrease_audio_speed(self, event=None):
        """Decrease reference audio playback speed"""
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
        self.update_audio_speed_display()
        print(f"Audio speed: {self.playback_speed:.1f}x")

    def increase_audio_speed(self, event=None):
        """Increase reference audio playback speed"""
        self.playback_speed = min(5.0, self.playback_speed + 0.25)
        self.update_audio_speed_display()
        print(f"Audio speed: {self.playback_speed:.1f}x")

    def update_audio_speed_display(self):
        """Update audio speed display"""
        if hasattr(self, 'audio_speed_display'):
            self.audio_speed_display.set_text(f'Audio Speed: {self.playback_speed:.1f}x')

    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'up' or event.key == '+':
            self.increase_sai_speed()
        elif event.key == 'down' or event.key == '-':
            self.decrease_sai_speed()
        elif event.key == 'right':
            self.increase_audio_speed()
        elif event.key == 'left':
            self.decrease_audio_speed()
        elif event.key == 'r':
            self.sai_speed = 1.0
            self.playback_speed = 1.0
            self.update_sai_speed_display()
            self.update_audio_speed_display()
            print("Both speeds reset to 1.0x")

    def calculate_similarity(self, detected_phonemes, target_phonemes):
        """Calculate similarity between detected and target phonemes"""
        if not target_phonemes or not detected_phonemes:
            return 0.0, "0.0%"
        
        # Clean up strings
        detected = detected_phonemes.lower().strip()
        target = target_phonemes.lower().strip()
        
        # Simple character overlap similarity
        if len(target) == 0:
            return 0.0, "0.0%"
        
        # Count matching characters
        common_chars = 0
        for char in detected:
            if char in target:
                common_chars += 1
        
        # Calculate similarity as percentage
        max_len = max(len(detected), len(target))
        if max_len == 0:
            similarity = 1.0
        else:
            similarity = common_chars / max_len
        
        percentage = f"{similarity * 100:.1f}%"
        return similarity, percentage

    def _load_audio_file(self):
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, original_sr = librosa.load(self.audio_file_path, sr=None)
        
        if original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
        
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        
        # Auto-detect content
        filename = os.path.basename(self.audio_file_path).lower()
        if 'thank' in filename:
            self.set_reference_text('sje sje', 'xiè xiè', 'thank you')
        elif 'hello' in filename:
            self.set_reference_text('ni hao', 'nǐ hǎo', 'hello')
        else:
            self.set_reference_text('sje sje', 'xiè xiè', 'thank you')
        
        if self.audio_playback_enabled:
            self._setup_audio_playback()

    def set_reference_text(self, phonemes, pronunciation, translation):
        self.reference_text = phonemes.strip()
        self.reference_pronunciation = pronunciation
        self.translated_text = translation.strip()
        print(f"Reference: {self.reference_text} ({self.reference_pronunciation}) - {self.translated_text}")
        print(f"Target phonemes for similarity: {self.target_phonemes}")

    def _setup_audio_playback(self):
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback
            )
            print("Audio playback stream created")
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _audio_playback_callback(self, outdata, frames, time, status):
        try:
            if self.audio_data is not None:
                start_pos = int(self.playback_position)
                end_pos = min(start_pos + frames, self.total_samples)

                # Calculate number of samples to read based on speed
                speed_factor = self.playback_speed
                chunk_indices = np.arange(frames) * speed_factor
                chunk_indices = chunk_indices.astype(int) + start_pos
                chunk_indices = np.clip(chunk_indices, 0, self.total_samples - 1)
                chunk = self.audio_data[chunk_indices]

                outdata[:, 0] = chunk

                self.playback_position += int(frames * speed_factor)
                if self.playback_position >= self.total_samples:
                    if self.loop_audio:
                        self.playback_position = 0
                    else:
                        outdata.fill(0)
            else:
                outdata.fill(0)
        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata.fill(0)

    def get_next_file_chunk(self):
        if self.audio_data is None:
            return None, -1
        
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
            else:
                return None, -1
        
        # Process at consistent rate regardless of playback speed
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        chunk = self.audio_data[self.current_position:end_position]
        
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        
        chunk_index = self.current_position
        
        # Advance position based on actual audio rate, not playback speed
        self.current_position = end_position
        
        return chunk.astype(np.float32), chunk_index

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for SAI visualization only (NOT for Wav2Vec2)"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to waveform buffer for display
            self.waveform_realtime.add_chunk(audio_float)
            
            try:
                self.audio_queue.put_nowait(audio_float)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_float)
                except queue.Empty:
                    pass
        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def process_realtime_audio(self):
        """Process real-time audio for SAI visualization only"""
        print("Real-time SAI processing started")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                nap_output = self.processor_realtime.process_chunk(audio_chunk)
                sai_output = self.sai_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)

                self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time processing error: {e}")
                continue

    def toggle_phoneme_recognition(self, event=None):
        if not self.wav2vec2_handler.is_recording:
            if self.wav2vec2_handler.start_recording():
                self.btn_transcribe.label.set_text('Stop Recognition')
                print("Started recording for phoneme recognition")

            if hasattr(self, 'similarity_display') and self.similarity_display is not None:
                self.similarity_display.set_text('')
            if hasattr(self, 'similarity_rect') and self.similarity_rect is not None:
                self.similarity_rect.remove()
                self.similarity_rect = None
                self.similarity_display = None
        else:
            self.wav2vec2_handler.stop_recording()
            self.btn_transcribe.label.set_text('Start Recognition')
            print("Stopped recording")

    def _setup_dual_visualization(self):
        print("DEBUG: Starting _setup_dual_visualization")
        self.fig = plt.figure(figsize=(16, 14))
        # Modified grid: SAI plots (8 rows) + waveforms (2 rows) + controls (2 rows)
        gs = self.fig.add_gridspec(12, 2, height_ratios=[1]*8 + [0.5, 0.5, 0.3, 0.3])
        
        # SAI plots
        self.ax_realtime = self.fig.add_subplot(gs[0:8, 0])
        self.ax_file = self.fig.add_subplot(gs[0:8, 1])
        
        # Waveform plots
        self.ax_waveform_realtime = self.fig.add_subplot(gs[8, 0])
        self.ax_waveform_file = self.fig.add_subplot(gs[8, 1])
        
        print("DEBUG: Created axes")
        
        # Setup SAI images
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_realtime.set_title("Live Microphone SAI", color='white', fontsize=14)
        self.ax_realtime.axis('off')
        
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        file_title = f"Reference: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file'}"
        self.ax_file.set_title(file_title, color='white', fontsize=14)
        self.ax_file.axis('off')
        
        # Setup waveform plots
        waveform_length = self.waveform_realtime.size
        time_axis = np.linspace(0, waveform_length / self.sample_rate, waveform_length)
        
        self.line_waveform_realtime, = self.ax_waveform_realtime.plot(
            time_axis, np.zeros(waveform_length), 'lime', linewidth=1
        )
        self.ax_waveform_realtime.set_xlim(0, waveform_length / self.sample_rate)
        self.ax_waveform_realtime.set_ylim(-1, 1)
        self.ax_waveform_realtime.set_title("Live Waveform", color='white', fontsize=10)
        self.ax_waveform_realtime.tick_params(colors='white', labelsize=8)
        self.ax_waveform_realtime.set_facecolor('black')
        
        self.line_waveform_file, = self.ax_waveform_file.plot(
            time_axis, np.zeros(waveform_length), 'cyan', linewidth=1
        )
        self.ax_waveform_file.set_xlim(0, waveform_length / self.sample_rate)
        self.ax_waveform_file.set_ylim(-1, 1)
        self.ax_waveform_file.set_title("Reference Waveform", color='white', fontsize=10)
        self.ax_waveform_file.tick_params(colors='white', labelsize=8)
        self.ax_waveform_file.set_facecolor('black')
        
        print("DEBUG: Created images and waveforms")
        
        # Text overlays
        print("DEBUG: Creating text overlays...")
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, 'Live SAI (separate phoneme recognition)', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=12, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_realtime")
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=12, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_file")
        
        self.transcription_status = self.ax_realtime.text(
            0.02, 0.12, 'Wav2Vec2 Phonemes: Click to start', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='orange', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        print("DEBUG: Created transcription_status")
        
        # Control area
        self.ax_controls = self.fig.add_subplot(gs[10:12, :])
        self.ax_controls.axis('off')

        # SAI Speed control buttons
        self.ax_sai_slower = plt.axes([0.02, 0.02, 0.08, 0.05])
        self.btn_sai_slower = Button(self.ax_sai_slower, 'SAI Slower', color='lightcoral', hovercolor='red')
        self.btn_sai_slower.on_clicked(self.decrease_sai_speed)

        self.ax_sai_faster = plt.axes([0.12, 0.02, 0.08, 0.05])
        self.btn_sai_faster = Button(self.ax_sai_faster, 'SAI Faster', color='lightblue', hovercolor='blue')
        self.btn_sai_faster.on_clicked(self.increase_sai_speed)

        # Audio speed buttons
        self.ax_audio_slower = plt.axes([0.22, 0.02, 0.08, 0.05])
        self.btn_audio_slower = Button(self.ax_audio_slower, 'Audio Slower', color='lightyellow', hovercolor='yellow')
        self.btn_audio_slower.on_clicked(self.decrease_audio_speed)

        self.ax_audio_faster = plt.axes([0.32, 0.02, 0.08, 0.05])
        self.btn_audio_faster = Button(self.ax_audio_faster, 'Audio Faster', color='lightgreen', hovercolor='green')
        self.btn_audio_faster.on_clicked(self.increase_audio_speed)
        
        # Main control buttons
        self.ax_playback = plt.axes([0.45, 0.02, 0.12, 0.05])
        self.btn_playback = Button(self.ax_playback, 'Play Reference', 
                                color='lightgreen', hovercolor='green')
        self.btn_playback.on_clicked(self.toggle_playback)
        
        self.ax_transcribe = plt.axes([0.60, 0.02, 0.12, 0.05])
        self.btn_transcribe = Button(self.ax_transcribe, 'Start Recognition', 
                                    color='lightblue', hovercolor='blue')
        self.btn_transcribe.on_clicked(self.toggle_phoneme_recognition)

        # Speed displays
        self.sai_speed_display = self.ax_controls.text(
            0.1, 0.8, f'SAI Speed: {self.sai_speed:.1f}x',
            transform=self.ax_controls.transAxes,
            fontsize=10, color='cyan', weight='bold'
        )
        
        self.audio_speed_display = self.ax_controls.text(
            0.1, 0.6, f'Audio Speed: {self.playback_speed:.1f}x',
            transform=self.ax_controls.transAxes,
            fontsize=10, color='yellow', weight='bold'
        )
        
        # Enable keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Use subplots_adjust instead of tight_layout to avoid compatibility issues
        try:
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12, hspace=0.3, wspace=0.3)
        except Exception as e:
            print(f"Layout adjustment warning: {e}")
        
        self.fig.patch.set_facecolor('black')
        print("DEBUG: Finished _setup_dual_visualization")

    def toggle_playback(self, event=None):
        if self.audio_playback_enabled and self.audio_output_stream:
            if self.audio_output_stream.active:
                self.audio_output_stream.stop()
                self.btn_playback.label.set_text('Play Reference')
                print("Reference playback stopped")
            else:
                self.audio_output_stream.start()
                self.btn_playback.label.set_text('Stop Reference')
                print("Reference playback started")
        else:
            print("Reference playback not available")

    def update_visualization(self, frame):
        try:
            # Update real-time SAI
            if self.vis_realtime.img is not None and self.vis_realtime.img.size > 0:
                current_max_rt = np.max(self.vis_realtime.img)
            else:
                current_max_rt = 1

            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))

            # Update real-time waveform
            waveform_data = self.waveform_realtime.get_waveform()
            self.line_waveform_realtime.set_ydata(waveform_data)

            try:
                detected = getattr(self.wav2vec2_handler, 'result', None)
                similarity = 0.0

                if detected and detected != "no_audio":
                    # FIXED: Only create similarity display if it doesn't exist
                    if not hasattr(self, 'similarity_display') or self.similarity_display is None:
                        self.similarity_rect = Rectangle(
                            (0.1, 0.3), 0.8, 0.4,
                            transform=self.ax_realtime.transAxes,
                            facecolor='black', alpha=0.6,
                            edgecolor='white', linewidth=2
                        )
                        self.ax_realtime.add_patch(self.similarity_rect)

                        self.similarity_display = self.ax_realtime.text(
                            0.5, 0.5, '',
                            transform=self.ax_realtime.transAxes,
                            verticalalignment='center',
                            horizontalalignment='center',
                            fontproperties=font_prop,
                            color='yellow', weight='bold', fontsize=18
                        )

                    # Update similarity text
                    if self.similarity_display is not None:
                        similarity_score, _ = self.calculate_similarity(detected, self.target_phonemes)
                        similarity = float(similarity_score) if similarity_score is not None else 0.0
                        similarity_percent = similarity * 100

                        # Updated scoring thresholds with feedback text
                        if similarity_percent >= 85:
                            self.similarity_display.set_color("lime")
                            feedback = "EXCELLENT!"
                        elif similarity_percent >= 70:
                            self.similarity_display.set_color("lightgreen")
                            feedback = "GREAT!"
                        elif similarity_percent >= 50:
                            self.similarity_display.set_color("orange")
                            feedback = "FAIR"
                        else:
                            self.similarity_display.set_color("red")
                            feedback = "PRACTICE MORE"

                        new_text = (
                            f"TARGET: {self.target_phonemes}\n"
                            f"DETECTED: {detected}\n"
                            f"SIMILARITY: {similarity_percent:.1f}% - {feedback}"
                        )
                        self.similarity_display.set_text(new_text)

            except Exception as e:
                print(f"DEBUG: Error updating similarity display: {e}")

            # Update file SAI visualization with speed control
            if self.audio_data is not None:
                chunk, chunk_index = self.get_next_file_chunk()
                if chunk is not None and chunk_index >= 0:
                    try:
                        # Add chunk to file waveform buffer
                        self.waveform_file.add_chunk(chunk)
                        
                        nap_output = self.processor_file.process_chunk(chunk)
                        sai_output = self.sai_file.RunSegment(nap_output)
                        self.vis_file.get_vowel_embedding(nap_output)
                        self.vis_file.run_frame(sai_output)

                        # SAI speed control (independent from audio speed)
                        self.sai_file_index += self.sai_speed
                        
                        if self.sai_file_index >= 1.0:
                            steps = int(self.sai_file_index)
                            self.sai_file_index -= steps
                            
                            for _ in range(steps):
                                if self.vis_file.img.shape[1] > 1:
                                    self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                                self.vis_file.draw_column(self.vis_file.img[:, -1])

                    except Exception as e:
                        print(f"Error processing file chunk: {e}")

            # Update file SAI display
            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))

            # Update file waveform
            file_waveform_data = self.waveform_file.get_waveform()
            self.line_waveform_file.set_ydata(file_waveform_data)

            # Update text displays
            reference_display = ''
            if self.reference_pronunciation:
                reference_display = f"Reference: {self.reference_pronunciation}"
            if self.translated_text and self.translated_text != 'audio file':
                reference_display += f" - {self.translated_text}"
            self.transcription_file.set_text(reference_display)

            status_text, status_color = self.wav2vec2_handler.get_current_status()
            self.transcription_status.set_text(status_text)
            self.transcription_status.set_color(status_color)

        except Exception as e:
            print(f"Visualization update error: {e}")

        # Return elements for FuncAnimation
        elements_to_return = [
            self.im_realtime, self.im_file,
            self.line_waveform_realtime, self.line_waveform_file,
            self.transcription_realtime, self.transcription_file,
            self.transcription_status
        ]
        if getattr(self, 'similarity_display', None) is not None:
            elements_to_return.append(self.similarity_display)

        return elements_to_return

    def start(self):
        print(f"Target phonemes: {self.target_phonemes}")
        
        # Initialize PyAudio for SAI visualization
        self.p = pyaudio.PyAudio()
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return

        self.running = True
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        # Start streams
        if self.stream:
            self.stream.start_stream()
            print("Audio input stream started (for SAI visualization)")
        
        if self.audio_playback_enabled and self.audio_output_stream:
            self.audio_output_stream.start()
            print("Reference audio playback started")
        
        # Start animation
        animation_interval = max(10, int((self.chunk_size / self.sample_rate) * 1000 / max(1, self.playback_speed)))
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        
        plt.show()

    def cleanup(self):
        self.running = False
        
        # Stop phoneme recognition
        if self.wav2vec2_handler.is_recording:
            self.wav2vec2_handler.stop_recording()
        
        if self.audio_output_stream:
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except:
                pass
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except:
            pass

    def stop(self):
        self.cleanup()
        plt.close('all')

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='SAI Visualization + Wav2Vec2 Phoneme Recognition + Similarity Scoring + Waveforms')
    parser.add_argument('--audio-file', default='reference/mandarin_thankyou.mp3', 
                        help='Path to reference audio file')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size for SAI')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width')
    parser.add_argument('--speed', type=float, default=3.0, help='Reference playback speed')
    parser.add_argument('--no-loop', action='store_true', help='Disable reference looping')
    parser.add_argument('--wav2vec2-model', default='facebook/wav2vec2-xlsr-53-espeak-cv-ft',
                        help='Wav2Vec2 model for phoneme recognition')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Warning: Audio file '{args.audio_file}' not found. Proceeding without reference audio.")
        args.audio_file = None

    # Initialize SAI visualization with Wav2Vec2
    sai_vis = SAIVisualizationWithWav2Vec2(
        audio_file_path=args.audio_file,
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        sai_width=args.sai_width,
        debug=args.debug,
        playback_speed=args.speed,
        loop_audio=not args.no_loop,
        wav2vec2_model=args.wav2vec2_model
    )

    try:
        sai_vis.start()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sai_vis.stop()
        print("Visualization stopped cleanly")

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_audio = os.path.join(script_dir, 'reference', 'mandarin_thankyou.mp3')
        
        if os.path.exists(default_audio):
            sys.argv.append('--audio-file')
            sys.argv.append(default_audio)
        else:
            print("No default audio file found. Starting with microphone SAI visualization only.")
    
    sys.exit(main() or 0)