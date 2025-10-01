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
from matplotlib.patches import FancyBboxPatch
import unicodedata

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

"""
# Configure matplotlib
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman', 'Arial Unicode MS', 'Segoe UI', 'sans-serif'],
    'axes.unicode_minus': False,
    'font.size': 12
})
"""

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

# ---------------- Visualization Handler ----------------
from modules.visualization_handler import VisualizationHandler, SAIParams

# ---------------- Phoneme Alignment and Feedback System ----------------
from modules.phoneme_handler import PhonemeHandler, PhonemeAnalyzer

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
    
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft", sample_rate=16000, target_phonemes="ɕiɛɕiɛ"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.enabled = True
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
        # Audio buffer for mic recording
        self.audio_buffer = []
        self.is_recording = False
        self.is_processing = False
        self.result = None

        # Phoneme analyzer setup
        self.target_phonemes = target_phonemes
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)
        
        # Add phoneme analysis results storage
        self.analysis_results = None
        self.overall_score = 0.0

        try:
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            self.recognizer = sr.Recognizer()
            with self.microphone:
                self.recognizer.adjust_for_ambient_noise(self.microphone)
            print("Microphone ready")
        except Exception as e:
            print(f"Microphone initialization warning: {e}")
            self.microphone = None
            self.recognizer = None

        # Load model immediately
        if not self.load_model():
            print("Failed to load Wav2Vec2 phoneme model. Handler disabled.")
            self.enabled = False

    def load_model(self):
        try:
            print(f"Loading Wav2Vec2 {self.model_name}...")
            
            # Load model and feature extractor first
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            
            print(f"Model loaded: {type(self.model)}")
            print(f"Feature extractor loaded: {type(self.feature_extractor)}")
            
            # Try to load a working tokenizer - try multiple approaches
            self.tokenizer = None
            
            # Approach 1: Try CTC tokenizer directly
            try:
                self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
                print(f"CTC Tokenizer loaded: {type(self.tokenizer)}")
                
                # Validate it's not a boolean
                if isinstance(self.tokenizer, bool):
                    print("CTC tokenizer returned boolean, trying alternatives...")
                    self.tokenizer = None
                else:
                    # Test the tokenizer
                    test_result = self.tokenizer.decode([1, 2, 3], skip_special_tokens=True)
                    print(f"CTC tokenizer test: '{test_result}'")
                    
            except Exception as e:
                print(f"CTC tokenizer failed: {e}")
                self.tokenizer = None
            
            # Approach 2: If CTC failed, try with a different model that definitely works
            if self.tokenizer is None or isinstance(self.tokenizer, bool):
                print("Trying alternative working model...")
                
                # Try a model that's known to work well with standard ASR
                working_model = "facebook/wav2vec2-base-960h"
                try:
                    print(f"Loading working model: {working_model}")
                    self.model = Wav2Vec2ForCTC.from_pretrained(working_model)
                    self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(working_model)
                    self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(working_model)
                    self.model_name = working_model
                    
                    print(f"Alternative model loaded successfully")
                    print(f"Alternative tokenizer: {type(self.tokenizer)}")
                    
                    # Test the alternative tokenizer
                    if not isinstance(self.tokenizer, bool):
                        test_result = self.tokenizer.decode([1, 2, 3], skip_special_tokens=True)
                        print(f"Alternative tokenizer test: '{test_result}'")
                    else:
                        print("Alternative tokenizer is also boolean!")
                        return False
                        
                except Exception as e:
                    print(f"Alternative model failed: {e}")
                    return False
            
            # Final validation
            if self.tokenizer is None or isinstance(self.tokenizer, bool):
                print("All tokenizer loading approaches failed")
                return False
            
            # Check if we have all required components
            if not all([self.model, self.feature_extractor, self.tokenizer]):
                print("Missing required components")
                return False
            
            # Try a comprehensive test
            try:
                print("Running comprehensive test...")
                
                # Create dummy audio
                test_audio = np.random.randn(1600).astype(np.float32)  # 0.1 seconds at 16kHz
                
                # Feature extraction
                inputs = self.feature_extractor(test_audio, sampling_rate=self.sample_rate, return_tensors="pt")
                print(f"Feature extraction successful: {inputs.input_values.shape}")
                
                # Model inference
                with torch.no_grad():
                    logits = self.model(inputs.input_values).logits
                print(f"Model inference successful: {logits.shape}")
                
                # Tokenization
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                print(f"Tokenization successful: '{decoded[0]}'")
                
            except Exception as test_error:
                print(f"Comprehensive test failed: {test_error}")
                return False
            
            print("Model loaded successfully with all components working")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def record(self, duration=3):
        """Record audio from microphone"""
        if not self.enabled:
            print("Handler not enabled")
            return None
        
        print(f"Recording {duration} seconds from microphone...")
        self.audio_buffer = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = self.audio_buffer.flatten()
        print("Recording finished")
        return audio

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

    def get_current_result(self):
        """Get the latest transcription result"""
        return self.result if self.result else "no_audio"

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
            
            # Process with manual components
            print(f"Processing with manual components")
            print(f"Model: {type(self.model)}")
            print(f"Feature extractor: {type(self.feature_extractor)}")
            print(f"Tokenizer: {type(self.tokenizer)}")
            
            try:
                # Manual processing step by step
                inputs = self.feature_extractor(
                    complete_audio, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                print(f"Input shape: {inputs.input_values.shape}")
                
                with torch.no_grad():
                    logits = self.model(inputs.input_values).logits
                
                print(f"Logits shape: {logits.shape}")
                
                predicted_ids = torch.argmax(logits, dim=-1)
                print(f"Predicted IDs shape: {predicted_ids.shape}")
                print(f"Sample predicted IDs: {predicted_ids[0][:20].tolist()}")
                
                # Decode the predictions
                transcription = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                print(f"Raw transcription: '{transcription}'")
                
                # Clean up the transcription
                cleaned_transcription = transcription.strip()
                
                if cleaned_transcription and len(cleaned_transcription.strip()) > 0:
                    self.result = cleaned_transcription  # Use the cleaned version
                    
                    # MOVED: Analyze pronunciation once here, after successful transcription
                    try:
                        self.analysis_results, self.overall_score = self.phoneme_analyzer.analyze_pronunciation(
                            cleaned_transcription, self.target_phonemes
                        )
                        print(f"Phoneme analysis completed. Overall score: {self.overall_score:.2f}")
                    except Exception as analysis_error:
                        print(f"Phoneme analysis error: {analysis_error}")
                        self.analysis_results = None
                        self.overall_score = 0.0
                        
                else:
                    self.result = "no_audio"
                    self.analysis_results = None
                    self.overall_score = 0.0
                    print("No transcription generated")
                    
            except Exception as processing_error:
                print(f"Processing error: {processing_error}")
                import traceback
                traceback.print_exc()
                self.result = "processing_error"
                self.analysis_results = None
                self.overall_score = 0.0
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            self.result = "no_audio"
            self.analysis_results = None
            self.overall_score = 0.0
        
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

# ---------------- Main SAI Visualization with Wav2Vec2 and Per-Phoneme Feedback ----------------
class SAIVisualizationWithWav2Vec2:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 debug=True, playback_speed=1.0, loop_audio=True, wav2vec2_model="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):

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
        self.target_phonemes = "ɕiɛɕiɛ"  # Target for xiè xiè

        # Initialize Wav2Vec2 handler and phoneme analyzer
        self.wav2vec2_handler = SimpleWav2Vec2Handler(model_name=wav2vec2_model)
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)

        # Most recent results of phoneme analysis
        
        # Feedback display components
        self.phoneme_feedback_displays = []
        self.feedback_background = None
        
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
        elif event.key == 'c':
            self.clear_phoneme_feedback()

    def clear_phoneme_feedback(self):
        """Clear all phoneme feedback displays"""
        for display in self.phoneme_feedback_displays:
            if display is not None:
                display.remove()
        self.phoneme_feedback_displays.clear()
        
        if self.feedback_background is not None:
            self.feedback_background.remove()
            self.feedback_background = None
        
        # print("Phoneme feedback cleared")

        # Enhanced color scheme with more granular feedback
    def create_phoneme_feedback_display(self, analysis_results, overall_score):
        """Create detailed per-phoneme feedback display with enhanced color coding"""
        try:
            # Clear existing displays
            self.clear_phoneme_feedback()
            
            if not analysis_results:
                return
            
            # Enhanced color scheme with more granular feedback
            def get_phoneme_color(result):
                """Get color based on phoneme accuracy with more granular levels"""
                status = result['status']
                similarity = result['similarity']
                
                if status == 'missing':
                    return '#666666'      # Dark gray for missing
                elif status == 'extra':
                    return '#FF00FF'      # Magenta for extra
                elif status == 'correct':
                    if similarity >= 0.95:
                        return '#00FF00'  # Bright green for perfect
                    else:
                        return '#44FF44'  # Slightly dimmer green for very good
                elif status == 'close':
                    if similarity >= 0.7:
                        return '#FFAA00'  # Orange for close
                    elif similarity >= 0.6:
                        return '#FFCC44'  # Yellow-orange for somewhat close
                    else:
                        return '#FFFF00'  # Yellow for barely acceptable
                else:  # incorrect
                    if similarity >= 0.3:
                        return '#FF6600'  # Red-orange for somewhat wrong
                    elif similarity >= 0.1:
                        return '#FF3300'  # Red for wrong
                    else:
                        return '#CC0000'  # Dark red for very wrong
            
            # Overall score display with color gradient
            def get_overall_score_color(score):
                """Get color for overall score with smooth gradient"""
                if score >= 0.9:
                    return '#00FF00'    # Bright green
                elif score >= 0.8:
                    return '#44FF44'    # Light green
                elif score >= 0.7:
                    return '#88FF00'    # Green-yellow
                elif score >= 0.6:
                    return '#FFFF00'    # Yellow
                elif score >= 0.5:
                    return '#FFAA00'    # Orange
                elif score >= 0.3:
                    return '#FF6600'    # Red-orange
                else:
                    return '#FF0000'    # Red
            
            score_color = get_overall_score_color(overall_score)
            overall_display = self.ax_realtime.text(
                0.5, 0.7,
                f"Overall Score: {overall_score*100:.0f}%",
                transform=self.ax_realtime.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16, fontweight='bold',
                color=score_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white')
            )
            self.phoneme_feedback_displays.append(overall_display)
            
            # Target phonemes display
            target_text = "TARGET: " + "".join([r['target'] or '∅' for r in analysis_results])
            target_display = self.ax_realtime.text(
                0.5, 0.62,
                target_text,
                transform=self.ax_realtime.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, fontweight='bold',
                color='cyan',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
            self.phoneme_feedback_displays.append(target_display)
            
            # Detected phonemes display with individual colors
            detected_text = "YOURS:  "
            detected_display = self.ax_realtime.text(
                0.1, 0.54,  # X position might need adjustment
                detected_text,
                transform=self.ax_realtime.transAxes,
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=14, fontweight='bold',
                color='white'
            )
            self.phoneme_feedback_displays.append(detected_display)
            
            # Individual phoneme displays with enhanced colors
            start_x = 0.25  # Increase this value to move phonemes further right
            x_step = 0.08   # Might need to be larger for wider characters
            
            for i, result in enumerate(analysis_results):
                x_pos = start_x + (i * x_step)
                
                # Get phoneme character and enhanced color
                phoneme_char = result['detected'] or '∅'
                color = get_phoneme_color(result)
                
                # Create individual phoneme display
                phoneme_display = self.ax_realtime.text(
                    x_pos, 0.54,
                    phoneme_char,
                    transform=self.ax_realtime.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3)
                )
                self.phoneme_feedback_displays.append(phoneme_display)
                
                # Add similarity percentage below phoneme for imperfect ones
                if result['similarity'] > 0 and result['similarity'] < 0.8:
                    similarity_text = f"{result['similarity']*100:.0f}%"
                    similarity_display = self.ax_realtime.text(
                        x_pos, 0.50,
                        similarity_text,
                        transform=self.ax_realtime.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color=color,
                        alpha=0.7
                    )
                    self.phoneme_feedback_displays.append(similarity_display)
            
            # Enhanced legend with more color categories
            legend_y = 0.42
            legend_items = [
                ("Perfect", '#00FF00'),
                ("Good", '#44FF44'),
                ("Close", '#FFAA00'),
                ("Poor", '#FF6600'),
                ("Wrong", '#FF0000'),
                ("Missing", '#666666')
            ]
            
            legend_start_x = 0.1
            x_spacing = 0.12
            
            for i, (label, color) in enumerate(legend_items):
                legend_display = self.ax_realtime.text(
                    legend_start_x + i * x_spacing, legend_y,
                    f"● {label}",
                    transform=self.ax_realtime.transAxes,
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=9, fontweight='bold',
                    color=color
                )
                self.phoneme_feedback_displays.append(legend_display)
            
            # Instructions
            instructions = "Press 'C' to clear feedback | Colors show pronunciation accuracy"
            instr_display = self.ax_realtime.text(
                0.5, 0.32,
                instructions,
                transform=self.ax_realtime.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9, style='italic',
                color='lightgray'
            )
            self.phoneme_feedback_displays.append(instr_display)
            
        except Exception as e:
            print(f"Error creating phoneme feedback display: {e}")
            import traceback
            traceback.print_exc()

    def calculate_similarity(self, detected_phonemes, target_phonemes):
        """Calculate similarity between detected and target phonemes"""
        if not target_phonemes or not detected_phonemes:
            return 0.0, "0.0%"
        
        # Use the phoneme analyzer for detailed analysis
        print(detected_phonemes, target_phonemes)
        analysis_results, overall_score = self.phoneme_analyzer.analyze_pronunciation(
            detected_phonemes, target_phonemes
        )
        
        percentage = f"{overall_score * 100:.1f}%"
        return overall_score, percentage

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
                # Clear previous feedback when starting new recording
                self.clear_phoneme_feedback()
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
        self.ax_realtime.set_title("Live Microphone SAI + Phoneme Feedback", color='white', fontsize=14)
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
            0.02, 0.02, 'Live SAI + Per-Phoneme Feedback', 
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
        
        # Clear feedback button
        self.ax_clear = plt.axes([0.75, 0.02, 0.08, 0.05])
        self.btn_clear = Button(self.ax_clear, 'Clear (C)', color='lightgray', hovercolor='gray')
        self.btn_clear.on_clicked(lambda x: self.clear_phoneme_feedback())

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
        
        # Keyboard shortcut help
        shortcuts_text = "Keys: ↑/↓ SAI speed, ←/→ audio speed, R reset, C clear feedback"
        self.ax_controls.text(
            0.5, 0.4, shortcuts_text,
            transform=self.ax_controls.transAxes,
            fontsize=9, color='lightgray', style='italic',
            horizontalalignment='center'
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

                if detected and detected != "no_audio":
                    # Get the pre-computed analysis results instead of calling analyze_pronunciation
                    analysis_results = getattr(self.wav2vec2_handler, 'analysis_results', None)
                    overall_score = getattr(self.wav2vec2_handler, 'overall_score', 0.0)
                    
                    # Only create display if we have analysis results
                    if analysis_results is not None:
                        self.create_phoneme_feedback_display(analysis_results, overall_score)

            except Exception as e:
                print(f"DEBUG: Error updating phoneme feedback: {e}")

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
        
        # Add phoneme feedback displays to return list
        elements_to_return.extend([d for d in self.phoneme_feedback_displays if d is not None])
        
        if self.feedback_background is not None:
            elements_to_return.append(self.feedback_background)

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
        
        # Clear phoneme feedback displays
        self.clear_phoneme_feedback()
        
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
    parser = argparse.ArgumentParser(description='SAI Visualization + Wav2Vec2 Phoneme Recognition + Per-Phoneme Color Feedback + Waveforms')
    parser.add_argument('--audio-file', default='reference/mandarin_thankyou.mp3', 
                        help='Path to reference audio file')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size for SAI')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width')
    parser.add_argument('--speed', type=float, default=1.0, help='Reference playback speed')
    parser.add_argument('--no-loop', action='store_true', help='Disable reference looping')
    parser.add_argument('--wav2vec2-model', default='facebook/wav2vec2-xlsr-53-espeak-cv-ft',
                        help='Wav2Vec2 model for phoneme recognition')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Warning: Audio file '{args.audio_file}' not found. Proceeding without reference audio.")
        args.audio_file = None

    # Initialize SAI visualization with Wav2Vec2 and per-phoneme feedback
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