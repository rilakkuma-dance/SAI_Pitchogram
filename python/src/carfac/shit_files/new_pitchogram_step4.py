import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
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

script_dir = os.path.dirname(os.path.abspath(__file__))
default_audio = os.path.join(script_dir, 'reference', 'mandarin_thankyou.mp3')

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: whisper not found. Install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

# Try to import JAX/CARFAC
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

plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
    'Arial Unicode MS', 'Tahoma', 'Times New Roman', 'Calibri',
    'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class SAIParams:
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

# ---------------- Chunk-based Whisper Handler with Audio Saving ----------------
class SimpleWhisperHandler:
    """Simple Whisper handler - records all audio then transcribes once"""
    
    def __init__(self, model_name="medium"):
        self.model_name = model_name
        self.enabled = WHISPER_AVAILABLE
        
        if not self.enabled:
            print("Whisper not available")
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
        
        # Whisper model
        self.audio_model = None
        
        # Calibrate microphone
        try:
            with self.microphone:
                self.recognizer.adjust_for_ambient_noise(self.microphone)
            print("Microphone ready")
        except Exception as e:
            print(f"Microphone warning: {e}")
        
        # ADD THIS: Pre-load Whisper model during initialization
        print("Initializing Whisper model...")
        if not self.load_model():
            print("Warning: Failed to load model during initialization")

    # ADD THIS METHOD:
    def load_model(self):
        """Load Whisper model"""
        if not self.enabled:
            print("Whisper not enabled, skipping model load")
            return False
        
        if self.audio_model is None:
            try:
                print(f"Loading Whisper {self.model_name}...")
                self.audio_model = whisper.load_model(self.model_name)
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
            return "Whisper not available", "red"
        
        if self.is_recording:
            return "Recording...", "yellow"
        
        if self.is_processing:
            return "Transcribing...", "orange"
        
        if self.result is not None:
            if self.result == "no_audio":
                return "No audio detected", "gray"
            else:
                return f"Transcript: {self.result}", "green"
        
        return "Ready to start", "blue"

    def start_recording(self):
        """Start recording"""
        if not self.enabled or self.is_recording:
            return False
        
        # Model should already be loaded during init
        if self.audio_model is None:
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
                while self.is_recording:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                        self.audio_data.append(audio_data)
                    except sr.WaitTimeoutError:
                        continue
        except Exception as e:
            print(f"Recording error: {e}")

    def _process_audio(self):
        """Process all recorded audio"""
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
            
            # Transcribe
            result = self.audio_model.transcribe(filename, language=None)
            text = result['text'].strip()
            
            if text and len(text) > 1:
                self.result = text
                print(f"Transcribed: {text}")
            else:
                self.result = "no_audio"
                print("No speech detected")
                
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

    # Add this to your SimpleWhisperHandler class:

    from difflib import SequenceMatcher

    def compare_with_reference(self, reference_text):
        """
        Compare the spoken result with a given reference text.
        Returns a dict with status, match boolean, similarity score, and texts.
        """
        if not hasattr(self, 'result') or self.result is None:
            return {'status': 'no_audio', 'reference': reference_text, 'spoken': ''}

        spoken_text = self.result.get('text', '').strip()
        if not spoken_text:
            return {'status': 'no_audio', 'reference': reference_text, 'spoken': ''}

        # Clean up both strings (lowercase, remove extra spaces)
        ref_clean = reference_text.lower().strip()
        spoken_clean = spoken_text.lower().strip()

        # Similarity ratio (0.0 - 1.0)
        similarity = SequenceMatcher(None, ref_clean, spoken_clean).ratio()

        # Decide if it’s a match (tweak threshold as needed)
        is_match = similarity >= 0.7  

        return {
            'status': 'comparison_ready',
            'match': is_match,
            'similarity': similarity,
            'reference': reference_text,
            'spoken': spoken_text
        }


    # Add this new method to your main class:

    def update_comparison_display(self):
        """Update the comparison display with text and visual feedback"""
        if not hasattr(self, 'comparison_text'):
            return
        
        # Get text comparison from Whisper
        if self.reference_text:
            comparison = self.whisper_handler.compare_with_reference(self.reference_text)
            
            if comparison['status'] == 'comparison_ready':
                if comparison['match']:
                    # Good match - show success
                    match_color = 'lime' if comparison['similarity'] >= 0.9 else 'yellow'
                    comparison_text = f"✓ MATCH ({comparison['similarity']:.1%})\n"
                    comparison_text += f"Target: {comparison['reference']}\n"
                    comparison_text += f"Spoken: {comparison['spoken']}"
                else:
                    # Poor match - show difference
                    match_color = 'red'
                    comparison_text = f"✗ NO MATCH ({comparison['similarity']:.1%})\n"
                    comparison_text += f"Target: {comparison['reference']}\n"
                    comparison_text += f"Spoken: {comparison['spoken']}"
            elif comparison['status'] == 'no_audio':
                match_color = 'gray'
                comparison_text = f"Target: {comparison['reference']}\nSpoken: {comparison['spoken']}"
            else:
                match_color = 'white'
                comparison_text = "Ready for comparison"
        else:
            match_color = 'white'
            comparison_text = "No reference text set"
        
        self.comparison_text.set_text(comparison_text)
        self.comparison_text.set_color(match_color)

        # Update your _setup_dual_visualization method to include comparison display:

# Replace your current _setup_dual_visualization method with this:
    def _setup_dual_visualization(self):
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(12, 2, height_ratios=[1]*8 + [0.5, 0.5, 0.3, 0.3])
        
        # SAI visualizations (top 8 rows)
        self.ax_realtime = self.fig.add_subplot(gs[0:8, 0])
        self.ax_file = self.fig.add_subplot(gs[0:8, 1])
        
        # Setup SAI images
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_realtime.set_title("Your Pronunciation (SAI)", color='white', fontsize=14)
        self.ax_realtime.axis('off')
        
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        file_title = f"Reference: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file'}"
        self.ax_file.set_title(file_title, color='white', fontsize=14)
        self.ax_file.axis('off')
        
        # Comparison area (rows 8-9)
        self.ax_comparison = self.fig.add_subplot(gs[8:10, :])
        self.ax_comparison.axis('off')
        self.ax_comparison.set_facecolor('black')
        
        # Text comparison display
        self.comparison_text = self.ax_comparison.text(
            0.5, 0.5, 'Dual Comparison: Text (Whisper) + Visual (SAI)\nSpeak and compare!', 
            transform=self.ax_comparison.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Status displays (row 10)
        status_ax_left = self.fig.add_subplot(gs[10, 0])
        status_ax_left.axis('off')
        self.transcription_status = status_ax_left.text(
            0.5, 0.5, 'Ready to start', 
            transform=status_ax_left.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='blue', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        status_ax_right = self.fig.add_subplot(gs[10, 1])
        status_ax_right.axis('off')
        self.reference_display = status_ax_right.text(
            0.5, 0.5, '', 
            transform=status_ax_right.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Control buttons (row 11)
        self.ax_controls = self.fig.add_subplot(gs[11, :])
        self.ax_controls.axis('off')
        
        # Buttons
        self.ax_playback = plt.axes([0.1, 0.02, 0.15, 0.05])
        self.btn_playback = Button(self.ax_playback, 'Play Reference', 
                                color='lightgreen', hovercolor='green')
        self.btn_playback.on_clicked(self.toggle_playback)
        
        self.ax_transcribe = plt.axes([0.3, 0.02, 0.15, 0.05])
        self.btn_transcribe = Button(self.ax_transcribe, 'Start Recording', 
                                    color='lightblue', hovercolor='blue')
        self.btn_transcribe.on_clicked(self.toggle_chunk_transcription)
        
        self.ax_compare = plt.axes([0.5, 0.02, 0.15, 0.05])
        self.btn_compare = Button(self.ax_compare, 'Compare Now', 
                                color='orange', hovercolor='darkorange')
        self.btn_compare.on_clicked(self.manual_compare)
        
        self.ax_reset = plt.axes([0.7, 0.02, 0.15, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Reset', 
                            color='lightcoral', hovercolor='red')
        self.btn_reset.on_clicked(self.reset_comparison)
        
        plt.tight_layout()
        self.fig.patch.set_facecolor('black')
    # Add these new methods to your main class:

    def manual_compare(self, event=None):
        """Manually trigger comparison"""
        self.update_comparison_display()
        print("Comparison updated")

    def reset_comparison(self, event=None):
        """Reset the comparison and clear results"""
        self.whisper_handler.result = None
        self.whisper_handler.is_processing = False
        self.whisper_handler.is_recording = False
        
        # Reset button text
        self.btn_transcribe.label.set_text('Start Recording')
        
        # Reset comparison display
        if hasattr(self, 'comparison_text'):
            self.comparison_text.set_text('Dual Comparison: Text (Whisper) + Visual (SAI)\nSpeak and compare!')
            self.comparison_text.set_color('white')
        
        print("Comparison reset")

    # Update your update_visualization method to include comparison updates:

    def update_visualization(self, frame):
        try:
            # ... existing SAI visualization code ...
            
            # Update status displays
            status_text, status_color = self.whisper_handler.get_current_status()
            self.transcription_status.set_text(status_text)
            self.transcription_status.set_color(status_color)
            
            # Update reference display
            reference_display = ''
            if self.reference_text and self.reference_pronunciation:
                reference_display = f"{self.reference_text} ({self.reference_pronunciation})"
            elif self.reference_text:
                reference_display = self.reference_text
            if self.translated_text:
                reference_display += f"\n{self.translated_text}"
            self.reference_display.set_text(reference_display)
            
            # Auto-update comparison when transcription is complete
            if (hasattr(self.whisper_handler, 'result') and 
                self.whisper_handler.result is not None and 
                not self.whisper_handler.is_processing and 
                not self.whisper_handler.is_recording):
                self.update_comparison_display()
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.im_file, self.comparison_text, 
                self.transcription_status, self.reference_display]
    
# Minimal fallback audio processor (very simple NAP-like output)
class SimpleAudioProcessor:
    def __init__(self, fs=16000, n_channels=200):
        self.fs = fs
        self.n_channels = n_channels

    def process_chunk(self, audio_chunk):
        # audio_chunk: 1D numpy float32 [-1..1]
        # produce simple NAP-like output: energy per band over short windows
        chunk = audio_chunk.astype(np.float32)
        if chunk.size == 0:
            return np.zeros((self.n_channels, 1), dtype=np.float32)
        # create a simple filterbank-ish division by downsampling windows
        length = max(1, len(chunk) // 32)
        energies = []
        for i in range(self.n_channels):
            start = (i * length) % max(1, len(chunk) - length + 1)
            seg = chunk[start:start + length]
            energies.append(np.mean(np.abs(seg)))
        # return shape (n_channels, 1) to mimic NAP channel x time
        return np.array(energies, dtype=np.float32).reshape(self.n_channels, 1)

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

# ---------------- Main SAI Visualization with Chunk Whisper ----------------
class SAIVisualizationWithChunkWhisper:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 debug=True, playback_speed=3.0, loop_audio=True, whisper_model="medium"):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        
        # Reference text
        self.reference_text = None
        self.reference_pronunciation = None
        self.translated_text = None
        
        # Initialize chunk-based Whisper handler
        self.whisper_handler = SimpleWhisperHandler(model_name=whisper_model)
        
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
        
        self._setup_dual_visualization()

    def _load_audio_file(self):
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, original_sr = librosa.load(self.audio_file_path, sr=None, mono=True)

        # Resample to match Whisper and SAI pipeline
        if original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=original_sr, target_sr=self.sample_rate)

        # Normalize
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.95

        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        self.current_position = 0
        print(f"File loaded: {self.total_samples} samples, duration {self.duration:.2f}s")
        
        # Auto-detect content
        filename = os.path.basename(self.audio_file_path).lower()
        if 'thank' in filename:
            self.set_reference('谢谢', 'xièxiè', 'thank you')
        elif 'hello' in filename:
            self.set_reference('你好', 'nǐhǎo', 'hello')
        else:
            self.set_reference('谢谢', 'xièxiè', 'thank you')
        
        if self.audio_playback_enabled:
            self._setup_audio_playback()

    def set_reference(self, text, pronunciation=None, translation=None):
        """
        Set the reference word/sentence along with optional pronunciation and translation.
        Updates the display immediately if visualization is active.
        """
        self.reference_text = text
        self.reference_pronunciation = pronunciation
        self.translated_text = translation

        # Update the display right away if the figure is active
        if hasattr(self, 'reference_display'):
            reference_display = text
            if pronunciation:
                reference_display += f" ({pronunciation})"
            if translation:
                reference_display += f"\n{translation}"

            self.reference_display.set_text(reference_display)
            self.reference_display.set_color('cyan')

        print(f"Reference set: {text}, pronunciation={pronunciation}, translation={translation}")

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
                
                if start_pos < self.total_samples:
                    chunk = self.audio_data[start_pos:end_pos]
                    
                    if len(chunk) < frames:
                        if self.loop_audio:
                            remaining = frames - len(chunk)
                            if remaining > 0:
                                loop_chunk = self.audio_data[:min(remaining, self.total_samples)]
                                chunk = np.concatenate([chunk, loop_chunk])
                                self.playback_position = len(loop_chunk)
                            else:
                                self.playback_position = 0
                        else:
                            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
                            self.playback_position = end_pos
                    else:
                        self.playback_position = end_pos
                    
                    if self.playback_position >= self.total_samples:
                        self.playback_position = 0
                    
                    outdata[:, 0] = chunk[:frames]
                else:
                    outdata.fill(0)
            else:
                outdata.fill(0)
        except Exception as e:
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
        
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        chunk = self.audio_data[self.current_position:end_position]
        
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        
        chunk_index = self.current_position
        self.current_position = end_position
        
        return chunk.astype(np.float32), chunk_index

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for SAI visualization only (NOT for Whisper)"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
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

    def toggle_chunk_transcription(self, event=None):
        if not self.whisper_handler.is_recording:
            if self.whisper_handler.start_recording():  # SimpleWhisperHandler method
                self.btn_transcribe.label.set_text('Stop Transcription')
                print("Started recording")  # Should say "Started recording", not "chunk-based"
        else:
            self.whisper_handler.stop_recording()  # SimpleWhisperHandler method
            self.btn_transcribe.label.set_text('Start Transcription')
            print("Stopped recording")  # Should say "Stopped recording"

    def _setup_dual_visualization(self):
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(12, 2, height_ratios=[1]*10 + [0.3, 0.3])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:10, 0])
        self.ax_file = self.fig.add_subplot(gs[0:10, 1])
        
        # Setup images
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
        
        # Text overlays
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, 'Live SAI (separate chunk transcription)', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=12, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=12, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        self.transcription_status = self.ax_realtime.text(
            0.02, 0.12, 'Chunk-based Whisper: Click to start', 
            transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='orange', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Control area
        self.ax_controls = self.fig.add_subplot(gs[10:12, :])
        self.ax_controls.axis('off')
        
        # Buttons
        self.ax_playback = plt.axes([0.25, 0.02, 0.15, 0.05])
        self.btn_playback = Button(self.ax_playback, 'Play Reference', 
                                  color='lightgreen', hovercolor='green')
        self.btn_playback.on_clicked(self.toggle_playback)
        
        self.ax_transcribe = plt.axes([0.6, 0.02, 0.15, 0.05])
        self.btn_transcribe = Button(self.ax_transcribe, 'Start Transcription', 
                                    color='lightblue', hovercolor='blue')
        self.btn_transcribe.on_clicked(self.toggle_chunk_transcription)
        
        plt.tight_layout()
        self.fig.patch.set_facecolor('black')

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
            # Update real-time SAI visualization
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Update file SAI visualization
            if self.audio_data is not None:
                chunk, chunk_index = self.get_next_file_chunk()
                if chunk is not None and chunk_index >= 0:
                    try:
                        nap_output = self.processor_file.process_chunk(chunk)
                        sai_output = self.sai_file.RunSegment(nap_output)
                        self.vis_file.get_vowel_embedding(nap_output)
                        self.vis_file.run_frame(sai_output)

                        self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                        self.vis_file.draw_column(self.vis_file.img[:, -1])
                    except Exception as e:
                        print(f"Error processing file chunk: {e}")

            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
            
            # Update text displays
            reference_display = ''
            if self.reference_text and self.reference_pronunciation:
                reference_display = f"{self.reference_text}({self.reference_pronunciation})"
            elif self.reference_text:
                reference_display = f"{self.reference_text}"
            
            if self.translated_text:
                reference_display += f"\n{self.translated_text}"
            
            self.transcription_file.set_text(reference_display)
            
            # Update transcription status with real-time workflow steps
            # Simple status update
            status_text, status_color = self.whisper_handler.get_current_status()
            self.transcription_status.set_text(status_text)
            self.transcription_status.set_color(status_color)
                        
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.im_file, self.transcription_realtime, 
                self.transcription_file, self.transcription_status]

    def start(self):
        print("Starting SAI Visualization with Chunk-based Whisper")
        print("SAI uses continuous microphone input for visualization")
        print("Whisper uses separate chunk-based recording for quality transcription")
        
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
        
        print("System Ready:")
        print("1. SAI visualization shows live microphone input")
        print("2. Click 'Start Transcription' for chunk-based Whisper recording")
        print("3. Two separate audio streams: SAI (continuous) + Whisper (chunks)")
        
        plt.show()

    def cleanup(self):
        self.running = False
        
        # Stop chunk-based transcription
        if self.whisper_handler.is_recording:
            self.whisper_handler.stop_recording()
        
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
# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='SAI Visualization + Chunk-based Whisper Transcription')
    parser.add_argument('--audio_file', type=str, default=None, help='Path to reference audio file')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Audio chunk size for SAI')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--sai_width', type=int, default=200, help='SAI width for visualization')
    parser.add_argument('--whisper_model', type=str, default='medium', help='Whisper model size')
    parser.add_argument('--playback_speed', type=float, default=1.0, help='Reference playback speed')
    parser.add_argument('--loop_audio', action='store_true', help='Loop reference audio')
    
    args = parser.parse_args()
    
    try:
        sai_whisper_system = SAIVisualizationWithChunkWhisper(
            audio_file_path=args.audio_file,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            playback_speed=args.playback_speed,
            loop_audio=args.loop_audio,
            whisper_model=args.whisper_model
        )
        
        sai_whisper_system.start()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
        sai_whisper_system.stop()
    except Exception as e:
        print(f"Error running the system: {e}")
        sai_whisper_system.stop()

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"Audio file {audio_file} not found. Exiting.")
            sys.exit(1)
        print(f"Using audio file: {audio_file}")
    else:
        if os.path.exists(default_audio):
            audio_file = default_audio
            print(f"No audio file provided. Using default audio: {default_audio}")
        else:
            audio_file = None
            print("No default audio file found. Starting with microphone SAI visualization only.")

if __name__ == '__main__':
    main()