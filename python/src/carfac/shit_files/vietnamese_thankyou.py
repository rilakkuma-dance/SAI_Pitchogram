import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
## from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import threading
import queue
import time
import torch
import whisper
from dataclasses import dataclass
import librosa
import argparse
import os
import editdistance

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac.jax.carfac as carfac

from carfac.np.carfac import CarParams
import sai

# Put Thai fonts first in the list
plt.rcParams['font.sans-serif'] = [
    # Thai fonts (put these FIRST)
    'Tahoma', 'Leelawadee UI', 'Cordia New', 'Angsana New', 'TH Sarabun New', 'Noto Sans Thai',
    # Then other fonts
    'Arial Unicode MS', 
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Times New Roman', 'Calibri',
    'DejaVu Sans', 'Liberation Sans', 'sans-serif'
]

# This is crucial for Thai text
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus'] = False  # Handle minus signs properly

@dataclass
class SAIParams:
    """SAI Parameters"""
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

    def __dict__(self):
        return {
            "num_channels": self.num_channels,
            "sai_width": self.sai_width,
            "future_lags": self.future_lags,
            "num_triggers_per_frame": self.num_triggers_per_frame,
            "trigger_window_width": self.trigger_window_width,
            "input_segment_width": self.input_segment_width,
            "channel_smoothing_scale": self.channel_smoothing_scale,
        }

@dataclass
class PitchogramParams:
    log_lag: bool = True
    lags_per_octave: float = 36.0
    min_lag_s: float = 0.0005
    log_offset_s: float = 0.0025
    vowel_time_constant_s: float = 0.02
    light_theme: bool = False

# ---------------- Similarity Calculator ----------------
class SimilarityCalculator:
    """Calculates similarity scores between recorded voice and reference file SAI"""
    def __init__(self, history_size=50, smoothing_factor=0.3):
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        
        self.file_sai_history = []  # Store file SAI frames for comparison
        self.recorded_sai_frames = []  # Store recorded SAI frames
        
        self.current_similarity = 0.0
        self.smoothed_similarity = 0.0
        self.max_similarity = 0.0
        self.similarity_history = []
        
    def add_file_frame(self, sai_frame):
        """Add a file SAI frame to history"""
        if len(self.file_sai_history) >= self.history_size:
            self.file_sai_history.pop(0)
        self.file_sai_history.append(sai_frame.copy())
    
    def set_recorded_frames(self, recorded_frames):
        """Set the recorded SAI frames for comparison"""
        self.recorded_sai_frames = recorded_frames

class TextSimilarityCalculator:
    """Calculates text similarity between recorded transcription and reference file transcription"""
    def __init__(self, file_transcription: str):
        self.file_transcription = self.preprocess_text(file_transcription)
        self.last_score_percentage = 0.0
        self.score_history = []

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by lowercasing, stripping whitespace, and removing punctuation"""
        text = text.lower().strip().strip(".,!?")
        return text

    def compare_texts(self, recorded_text: str) -> int:
        recorded_text = self.preprocess_text(recorded_text)

        if not recorded_text or not self.file_transcription:
            return 0

        distance = editdistance.eval(recorded_text, self.file_transcription)
        return distance

# ---------------- CARFAC Processor ----------------
class RealCARFACProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
            carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        )
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])

    def process_chunk(self, audio_chunk):
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)
        else:
            audio_input = audio_chunk
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        naps, _, self.state, _, _, _ = self.run_segment_jit(audio_jax, self.hypers, self.weights, self.state, open_loop=False)
        return np.array(naps[:, :, 0]).T

# ---------------- Whisper Handler ----------------
class WhisperHandler:
    def __init__(self, model_name="base", non_english=True, debug=True):
        self.debug = debug
        model = model_name
        if model_name != "large" and not non_english:
            model = model + ".en"
        
        try:
            self.audio_model = whisper.load_model(model)
            self.sample_rate = 16000
        except Exception as e:
            try:
                self.audio_model = whisper.load_model("large")
                self.sample_rate = 16000
            except Exception as e2:
                self.audio_model = None
                self.sample_rate = 16000
        
        self.transcription = []
        self.lock = threading.Lock()
        self.last_transcription_time = time.time()
        self.min_transcription_interval = 0.1

    def transcribe_audio(self, audio_data, language='en'):
        min_samples = int(self.sample_rate * 0.5)
        
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        
        target_length = max(min_samples, len(audio_float))
        if len(audio_float) < target_length:
            audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
        elif len(audio_float) > self.sample_rate * 30:
            audio_float = audio_float[-self.sample_rate * 30:]
        
        try:
            result = self.audio_model.transcribe(
                audio_float, 
                fp16=torch.cuda.is_available(),
                language=language,
                condition_on_previous_text=False,
            )
            text = result.get('text', '').strip()
            
            if len(text) < 1:
                return None
                
            return text
        except Exception:
            return None

    def add_transcription_line(self, text):
        with self.lock:
            if text is None or not text.strip():
                return
            
            current_time = time.time()
            if current_time - self.last_transcription_time < self.min_transcription_interval:
                return
            
            if self.transcription and self.transcription[-1] == text:
                return
                
            self.transcription.append(text)
            self.last_transcription_time = current_time
            print(f"[Transcribed]: {text}")
            
            if len(self.transcription) > 20:
                self.transcription = self.transcription[-20:]
    
    def get_last_line(self) -> str:
        with self.lock:
            return self.transcription[-1] if self.transcription else ""

    def get_display_text(self, max_lines=5, max_chars=200):
        with self.lock:
            if not self.transcription:
                return ""
            
            lines = self.transcription[-max_lines:]
            display = '\n'.join(lines)
            
            if len(display) > max_chars:
                display = "..." + display[-(max_chars-3):]
            
            return display
        
    def transcribe_once(self, audio_data, language='en'):
        """Transcribe audio data once and return the result immediately"""
        min_samples = int(self.sample_rate * 0.5)
        
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        
        target_length = max(min_samples, len(audio_float))
        if len(audio_float) < target_length:
            audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
        elif len(audio_float) > self.sample_rate * 30:
            audio_float = audio_float[-self.sample_rate * 30:]
        
        try:
            if self.debug:
                print("Starting transcription...")
            
            result = self.audio_model.transcribe(
                audio_float, 
                fp16=torch.cuda.is_available(),
                language=language,
                condition_on_previous_text=False,
                verbose=False,  # Reduce output noise
            )
            text = result.get('text', '').strip()
            
            if self.debug:
                print(f"Transcription completed: {text}")
            
            return text if len(text) > 0 else None
                
        except Exception as e:
            if self.debug:
                print(f"Transcription error: {e}")
            return None

# ---------------- Visualization Handler ----------------
class VisualizationHandler:
    @dataclass
    class ResamplingCell:
        left_index: int
        right_index: int
        left_weight: float
        interior_weight: float
        right_weight: float

        def __init__(self, left_edge: float, right_edge: float):
            cell_width: float = right_edge - left_edge

            if (cell_width < 1.0):
                grow: float = 0.5 * (1.0 - cell_width)
                left_edge -= grow
                right_edge += grow
            
            left_edge = max(0.0, left_edge)
            right_edge = max(0.0, right_edge)
            cell_width = right_edge - left_edge

            self.left_index = int(round(left_edge))
            self.right_index = int(round(right_edge))
            if (self.right_index > self.left_index and cell_width > 0.999):
                self.left_weight = (0.5 - (left_edge - self.left_index)) / cell_width
                self.interior_weight = 1.0 / cell_width
                self.right_weight = (0.5 + (right_edge - self.right_index)) / cell_width
            else:
                self.left_weight = 1.0
                self.interior_weight = 0.0
                self.right_weight = 0.0

        def CellAverage(self, samples: np.ndarray) -> float:
            if (self.left_index == self.right_index):
                return samples[self.left_index]
            return self.left_weight * samples[self.left_index] + \
                self.interior_weight * samples[self.left_index + 1 : self.right_index].sum() + \
                self.right_weight * samples[self.right_index]

    def __init__(self, sample_rate_hz: int, car_params: CarParams = CarParams(), sai_params: SAIParams = SAIParams(),
                 pitchogram_params: PitchogramParams = PitchogramParams()):
        self.temporal_buffer = np.zeros((sai_params.num_channels, sai_params.sai_width))
        self.car_params = car_params
        self.sai_params = sai_params
        self.pitchogram_params = pitchogram_params
        self.sample_rate_hz = sample_rate_hz
        self.workspace = np.zeros((sai_params.sai_width))
        self.output = np.zeros((sai_params.sai_width))
        self.pole_frequencies = self.car_pole_frequencies(sample_rate_hz, car_params)
        self.mask = np.ones((sai_params.num_channels, sai_params.sai_width), dtype=bool)
        center: int = sai_params.sai_width - sai_params.future_lags
        for c in range(self.pole_frequencies.shape[0]):
            half_cycle_samples: float = 0.5 * sample_rate_hz / self.pole_frequencies[c]
            i_start: int = int(np.clip(np.floor(center - half_cycle_samples), 0, sai_params.sai_width - 1))
            i_end: int = int(np.clip(np.floor(center + half_cycle_samples), 0, sai_params.sai_width - 1))
            self.mask[c, i_start:i_end+1] = 0
        self.vowel_matrix = self.create_vowel_matrix(sai_params.num_channels)
        self.vowel_coords = np.zeros((2, 1), dtype=np.float32)
        frame_rate_hz: float = sample_rate_hz / sai_params.input_segment_width
        self.cgram_smoother = 1 - np.exp(-1 / (pitchogram_params.vowel_time_constant_s * frame_rate_hz))
        self.cgram = np.zeros(self.pole_frequencies.shape, dtype=np.float32)
        self.log_lag_cells: list[VisualizationHandler.ResamplingCell] = list()
        if (not pitchogram_params.log_lag):
            self.output.resize((sai_params.sai_width))
        else:
            spacing: float = np.exp2(1.0 / pitchogram_params.lags_per_octave)
            log_offset: float = sample_rate_hz * pitchogram_params.log_offset_s
            left_edge: float = sample_rate_hz * pitchogram_params.min_lag_s
            while True:
                right_edge: float = (left_edge + log_offset) * spacing - log_offset
                cell: VisualizationHandler.ResamplingCell = self.ResamplingCell(left_edge, right_edge)
                if (cell.right_index >= sai_params.sai_width):
                    break
                self.log_lag_cells.append(cell)
                left_edge = right_edge
            self.workspace.resize((sai_params.sai_width))
            self.output.resize((len(self.log_lag_cells)))
        self.img = np.zeros((self.output.shape[0], 200, 3), dtype=np.uint8)

    def car_pole_frequencies(self, sample_rate_hz, car_params: CarParams) -> np.ndarray:
        num_channels: int = 0
        pole_hz: float = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        while pole_hz > car_params.min_pole_hz:
            num_channels += 1
            pole_hz -= car_params.erb_per_step * \
                ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)
        pole_freqs = np.zeros(num_channels, dtype=np.float32)
        pole_hz = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        for channel in range(num_channels):
            pole_freqs[channel] = pole_hz
            pole_hz -= car_params.erb_per_step * \
                ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)
        return pole_freqs
    
    def create_vowel_matrix(self, num_channels, erb_per_step=0.5) -> np.ndarray:
        def kernel(center, c):
            z = (c - center) / 3.3
            return np.exp((z * z) / -2)
        f2_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 2365)
        f2_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 1100)
        f1_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 700)
        f1_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 265)
        vowel_matrix = np.zeros((2, num_channels), dtype=np.float32)
        for c in range(num_channels):
            vowel_matrix[0, c] = kernel(f2_lo, c) - kernel(f2_hi, c)
            vowel_matrix[1, c] = kernel(f1_lo, c) - kernel(f1_hi, c)
        vowel_matrix *= erb_per_step / 2
        return vowel_matrix

    def frequency_to_channel_index(self, sample_rate_hz: int, erb_per_step: float, pole_freq: int):
        first_pole_theta: float = 0.85 * np.pi
        erb_q: float = 1000 / (24.7 * 4.37)
        pole0_hz: float = first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        break_freq: float = 165.3
        ratio: float = 1 - erb_per_step / erb_q
        min_pole_hz: float = 30
        pole_freq = np.clip(pole_freq, min_pole_hz, pole0_hz)
        top = np.log((pole_freq + break_freq) / (pole0_hz + break_freq))
        bottom = np.log(ratio)
        return top / bottom

    def get_vowel_embedding(self, nap) -> np.ndarray:
        self.cgram += self.cgram_smoother * (nap.mean(axis=1) - self.cgram)
        self.vowel_coords = self.vowel_matrix @ self.cgram
        return self.vowel_coords

    def run_frame(self, sai_frame: np.ndarray) -> np.ndarray:
        if (not self.pitchogram_params.log_lag):
            self.output = (sai_frame * self.mask).mean(axis=0)
        else:
            self.workspace = (sai_frame * self.mask).mean(axis=0)
            for i in range(self.output.shape[0]):
                self.output[i] = self.log_lag_cells[i].CellAverage(self.workspace)
        return self.output

    def draw_column(self, column_ptr: np.ndarray) -> None:
        v = np.ravel(self.vowel_coords)
        tint = np.array([
            0.5 - 0.6 * v[1],
            0.5 - 0.6 * v[0],
            0.35 * (v[0] + v[1]) + 0.4
        ], dtype=np.float32)
        k_scale: float = 0.5 * 255
        tint *= k_scale
        for i in range(self.output.shape[0]):
            column_ptr[i] = np.clip(np.int32((tint * self.output[i])), 0, 255)

# ---------------- Dual SAI with Recording ----------------
class DualSAIWithRecording:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
             whisper_model="base", whisper_interval=1.5, debug=True, playback_speed=3.0, 
             loop_audio=True, similarity_method='cosine', save_recordings=True, recording_dir="recordings",
             language: str = "vi"):
        
                # Audio saving functionality
        self.save_recordings = save_recordings
        self.recording_dir = recording_dir
        self.recording_counter = 0

        # Create recordings directory if it doesn't exist
        if self.save_recordings:
            os.makedirs(self.recording_dir, exist_ok=True)
            print(f"Recordings will be saved to: {os.path.abspath(self.recording_dir)}")

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_interval = whisper_interval
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        
        # Recording functionality
        self.is_recording = False
        self.recorded_audio = []
        self.recording_start_time = 0
        self.recording_duration = 5.0
        self.last_score_percentage = 0.0
        self.score_history = []
        
        # Initialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(history_size=50, smoothing_factor=0.2)
        self.similarity_method = similarity_method

        # defer this until we have a file transcription
        self.text_similarity_calculator: TextSimilarityCalculator
        
        # Initialize processing components for both sides
        self.carfac_realtime = RealCARFACProcessor(fs=sample_rate)
        self.carfac_file = RealCARFACProcessor(fs=sample_rate)
        self.n_channels = self.carfac_realtime.n_channels

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
        
        # Create separate SAI instances
        self.SAI_realtime = sai.SAI(self.sai_params)
        self.SAI_file = sai.SAI(self.sai_params)

        # Visualization handlers for both sides
        self.vis_realtime = VisualizationHandler(sample_rate, sai_params=self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, sai_params=self.sai_params)

        # Whisper handlers
        self.whisper_realtime = WhisperHandler(model_name=whisper_model, debug=debug)
        self.whisper_file = WhisperHandler(model_name=whisper_model, debug=debug)

        self.language: str = language

        # Real-time audio setup
        self.audio_queue = queue.Queue(maxsize=50)
        self.whisper_audio_buffer_realtime = []
        self.whisper_buffer_lock_realtime = threading.Lock()
        self.last_whisper_time_realtime = time.time()
        
        # File processing setup
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.original_sr = None
        self.current_position = 0
        self.chunks_processed = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0
        
        if audio_file_path and os.path.exists(audio_file_path):
            self._load_audio_file()
        
        self.whisper_audio_buffer_file = []
        self.whisper_buffer_lock_file = threading.Lock()
        self.last_whisper_time_file = 0

        # Audio stream
        self.p = None
        self.stream = None
        self.running = False
        
        # Setup visualization
        self._setup_dual_visualization()

    def _load_audio_file(self):
        """Load the audio file for file processing and transcribe it with Whisper"""
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, self.original_sr = librosa.load(self.audio_file_path, sr=None)
        
        if self.original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=self.original_sr, target_sr=self.sample_rate)
        
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate

        self.file_transcription = self.whisper_file.transcribe_audio(self.audio_data, language=self.language)
        self.whisper_file.add_transcription_line(self.file_transcription)

        self.text_similarity_calculator = TextSimilarityCalculator(self.file_transcription)

    def get_next_file_chunk(self):
        """Get next chunk from file with looping support"""
        if self.audio_data is None:
            return None, -1
        
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
                if self.debug:
                    print(f"Audio file looped (loop #{self.loop_count})")
            else:
                return None, -1
        
        chunk_index = int(self.current_position / self.chunk_size)
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        
        chunk = self.audio_data[self.current_position:end_position]
        
        if len(chunk) < self.chunk_size:
            if self.loop_audio and self.current_position + len(chunk) >= self.total_samples:
                remaining_samples = self.chunk_size - len(chunk)
                if remaining_samples > 0 and self.total_samples > 0:
                    loop_start = self.audio_data[:min(remaining_samples, self.total_samples)]
                    chunk = np.concatenate([chunk, loop_start])
                    self.current_position = len(loop_start)
                else:
                    self.current_position = end_position
            else:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                self.current_position = end_position
        else:
            self.current_position = end_position
        
        return chunk.astype(np.float32), chunk_index

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback with recording capability"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Handle recording
            if self.is_recording:
                self.recorded_audio.extend(audio_float)
                if len(self.recorded_audio) >= int(self.recording_duration * self.sample_rate):
                    self.stop_recording()
            
            try:
                self.audio_queue.put_nowait(audio_float)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_float)
                except queue.Empty:
                    pass

            # Add to Whisper buffer
            with self.whisper_buffer_lock_realtime:
                self.whisper_audio_buffer_realtime.extend(audio_float)
                max_buffer_size = self.sample_rate * 15
                if len(self.whisper_audio_buffer_realtime) > max_buffer_size:
                    excess = len(self.whisper_audio_buffer_realtime) - max_buffer_size
                    self.whisper_audio_buffer_realtime = self.whisper_audio_buffer_realtime[excess:]

        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def start_recording(self, event=None):
        """Start recording user's speech"""
        if self.is_recording:
            self.stop_recording()
            return
        
        self.is_recording = True
        self.recorded_audio = []
        self.recording_start_time = time.time()
        
        # Update button appearance
        self.btn_record.label.set_text('⏹ REC')
        self.btn_record.color = (1, 0, 0, 0.8)  # Red when recording
        
        # Start countdown thread
        threading.Thread(target=self._recording_countdown, daemon=True).start()

    def stop_recording(self, event=None):
        """Stop recording and calculate similarity score"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Update button appearance
        self.btn_record.label.set_text('🎤 REC')
        self.btn_record.color = (0, 0.8, 1, 0.8)  # Blue when not recording
        
        if len(self.recorded_audio) > 0:
            # Process recorded audio and calculate score
            threading.Thread(target=self._process_recorded_audio, daemon=True).start()
        else:
            print("No audio recorded")

    def _recording_countdown(self):
        """Handle recording countdown"""
        start_time = time.time()
        while self.is_recording and (time.time() - start_time) < self.recording_duration:
            remaining = self.recording_duration - (time.time() - start_time)
            if remaining > 0:
                time.sleep(0.1)
        
        if self.is_recording:
            self.stop_recording()

    def _process_recorded_audio(self):
        """Process recorded audio and compare against file SAI frames"""
        try:
            print("Processing recorded audio...")
            
            # Update status to show transcription is happening
            self.status_realtime.set_text("Transcribing...")
            self.status_realtime.set_color('yellow')
            
            # Convert to numpy array and ensure correct format
            recorded_array = np.array(self.recorded_audio, dtype=np.float32)
            
            if len(recorded_array) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                print("Recording too short for analysis")
                self.status_realtime.set_text("Recording too short")
                self.status_realtime.set_color('red')
                self.score_display.set_text("Recording too short - Please try again")
                self.score_display.set_color('red')
                return
            
            # Normalize audio
            if np.max(np.abs(recorded_array)) > 0:
                recorded_array = recorded_array / np.max(np.abs(recorded_array)) * 0.9
            
            # ONE-TIME transcription using the new method
            print("Transcribing recorded audio...")
            transcription = self.whisper_realtime.transcribe_once(recorded_array, language=self.language)
            
            # Update status back to ready
            self.status_realtime.set_text("Ready to record")
            self.status_realtime.set_color('white')
            
            if transcription and len(transcription.strip()) > 0:
                print(f"Recorded transcription: {transcription}")
                self.whisper_realtime.add_transcription_line(transcription)
                distance = self.text_similarity_calculator.compare_texts(transcription)
                
                if distance == 0:
                    self.transcription_realtime.set_color('lime')
                    self.score_display.set_text("PERFECT MATCH!")
                    self.score_display.set_color('lime')
                    print("Perfect match!")
                elif distance <= 1: 
                    self.transcription_realtime.set_color('lime')
                    self.score_display.set_text("EXCELLENT MATCH!")
                    self.score_display.set_color('lime')
                    print("Excellent match!")
                elif distance <= 3:
                    self.transcription_realtime.set_color('orange')
                    self.score_display.set_text(f"GOOD MATCH!")
                    self.score_display.set_color('orange')
                    print(f"Good match (distance: {distance})")
                else:
                    self.transcription_realtime.set_color('red')
                    self.score_display.set_text(f"NEEDS PRACTICE")
                    self.score_display.set_color('red')
                    print(f"Poor match (distance: {distance})")
            else:
                print("No transcription result")
                self.transcription_realtime.set_color('gray')
                self.score_display.set_text("No speech detected - Please try again")
                self.score_display.set_color('gray')
            
            # Process recorded audio through CARFAC and SAI (existing code)
            temp_carfac = RealCARFACProcessor(fs=self.sample_rate)
            temp_sai = sai.SAI(self.sai_params)
            
            # Split into chunks and process
            chunk_size = self.chunk_size
            recorded_sai_frames = []
            
            for i in range(0, len(recorded_array), chunk_size):
                chunk = recorded_array[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # Process chunk
                nap_output = temp_carfac.process_chunk(chunk)
                sai_output = temp_sai.RunSegment(nap_output)
                recorded_sai_frames.append(sai_output)
            
            # Store recorded frames in similarity calculator
            self.similarity_calculator.set_recorded_frames(recorded_sai_frames)
                    
        except Exception as e:
            self.status_realtime.set_text("Transcription failed")
            self.status_realtime.set_color('red')
            self.score_display.set_text("PROCESSING ERROR - Please try again")
            self.score_display.set_color('red')
            print(f"Error processing recorded audio: {e}")
            import traceback
            traceback.print_exc()

    def set_recording_duration(self, duration):
        """Set the recording duration in seconds"""
        self.recording_duration = max(1.0, min(10.0, duration))
        print(f"Recording duration set to {self.recording_duration:.1f} seconds")

    def process_realtime_audio(self):
        """Process real-time audio stream"""
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process with CARFAC
                nap_output = self.carfac_realtime.process_chunk(audio_chunk)
                # Process with SAI
                sai_output = self.SAI_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                pitch_frame = self.vis_realtime.run_frame(sai_output)

                # Update visualization
                self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])

            except Exception as e:
                print(f"Real-time audio processing error: {e}")
                continue

    def process_file_chunk_for_whisper(self, chunk, timestamp):
        pass

    def _process_whisper_file_chunk(self, audio_data, timestamp):
        """Process Whisper transcription for file"""
        try:
            text = self.whisper_file.transcribe_audio(audio_data, language=self.language)
            if text and len(text.strip()) > 0:
                loop_info = f" (Loop #{self.loop_count})" if self.loop_count > 0 and self.loop_audio else ""
                timestamped_text = f"[{timestamp:.1f}s{loop_info}] {text}"
                self.whisper_file.add_transcription_line(timestamped_text)
        except Exception as e:
            print(f"File Whisper processing error: {e}")

    def _setup_dual_visualization(self):
        """Setup dual screen visualization with recording controls"""
        self.fig = plt.figure(figsize=(20, 16))
        
        # Create layout
        gs = self.fig.add_gridspec(16, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.3, 0.3, 0.3, 0.3, 0.2])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:10, 0])
        self.ax_file = self.fig.add_subplot(gs[0:10, 1])
        
        # Score display area
        self.ax_score = self.fig.add_subplot(gs[11, :])
        self.ax_score.set_facecolor('black')
        self.ax_score.set_xlim(0, 1)
        self.ax_score.set_ylim(0, 1)
        self.ax_score.axis('off')
        
        # Score display text
        self.score_display = self.ax_score.text(
            0.5, 0.5, 'Press Record to test pronunciation', 
            transform=self.ax_score.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Control buttons area
        self.ax_controls = self.fig.add_subplot(gs[12:16, :])
        self.ax_controls.axis('off')
        
        # Enhanced colormap
        colors = ['#000022', '#000055', '#0033AA', '#0066FF', '#00AAFF',
                  '#00FFAA', '#33FF77', '#77FF33', '#AAFF00', '#FFAA00',
                  '#FF7700', '#FF3300', '#FF0044', '#CC0077', '#FFFFFF']
        self.cmap = LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)
        
        # Setup left side (real-time)
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_realtime.output.shape[0]]
        )
        self.ax_realtime.set_title("Real-time Microphone SAI", color='white', fontsize=14, pad=20)
        self.ax_realtime.axis('off')
        
        # Setup right side (file)
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_file.output.shape[0]]
        )
        file_title = f"Reference File SAI: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file loaded'}"
        self.ax_file.set_title(file_title, color='white', fontsize=14, pad=20)
        
        # Add text overlays for each side
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, '', transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=10, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Progress indicator for file
        self.progress_file = self.ax_file.text(
            0.02, 0.98, '', transform=self.ax_file.transAxes,
            verticalalignment='top', fontsize=10, color='yellow', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Status indicator for real-time
        self.status_realtime = self.ax_realtime.text(
            0.02, 0.98, '', transform=self.ax_realtime.transAxes,
            verticalalignment='top', fontsize=10, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Setup control buttons
        self._setup_control_buttons()
        
        plt.tight_layout()
        self.fig.patch.set_facecolor('black')

    def _setup_control_buttons(self):
        """Setup control buttons"""
        button_height = 0.05
        button_width = 0.08
        button_y = 0.02
        spacing = 0.12
        start_x = 0.2
        
        # Record button
        self.ax_record = plt.axes([start_x, button_y, button_width, button_height], 
                                 facecolor='none', frameon=False)
        self.btn_record = Button(self.ax_record, '🎤 REC', 
                                color=(0, 0.8, 1, 0.8), 
                                hovercolor=(0, 0.8, 1, 1.0))
        self.btn_record.label.set_fontsize(12)
        self.btn_record.label.set_color('white')
        self.btn_record.label.set_weight('bold')
        self.btn_record.on_clicked(self.start_recording)
        
        # Duration controls
        self.ax_duration_down = plt.axes([start_x + spacing, button_y, button_width*0.6, button_height], 
                                        facecolor='none', frameon=False)
        self.btn_duration_down = Button(self.ax_duration_down, '-', 
                                       color=(0.8, 0.8, 0.8, 0.6), 
                                       hovercolor=(0.8, 0.8, 0.8, 0.8))
        self.btn_duration_down.label.set_fontsize(16)
        self.btn_duration_down.label.set_color('white')
        self.btn_duration_down.label.set_weight('bold')
        self.btn_duration_down.on_clicked(lambda x: self.change_recording_duration(-1))
        
        # Duration display
        self.ax_duration_display = plt.axes([start_x + spacing + 0.08, button_y, button_width*0.8, button_height], 
                                           facecolor='none', frameon=False)
        self.duration_text = self.ax_duration_display.text(0.5, 0.5, f'{self.recording_duration:.1f}s', 
                                                          ha='center', va='center', 
                                                          fontsize=12, color='white', weight='bold',
                                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                                   facecolor=(0.2, 0.2, 0.2, 0.8), 
                                                                   edgecolor='white', alpha=0.8))
        self.ax_duration_display.set_xlim(0, 1)
        self.ax_duration_display.set_ylim(0, 1)
        self.ax_duration_display.axis('off')
        
        # Duration up button
        self.ax_duration_up = plt.axes([start_x + spacing + 0.16, button_y, button_width*0.6, button_height], 
                                      facecolor='none', frameon=False)
        self.btn_duration_up = Button(self.ax_duration_up, '+', 
                                     color=(0.8, 0.8, 0.8, 0.6), 
                                     hovercolor=(0.8, 0.8, 0.8, 0.8))
        self.btn_duration_up.label.set_fontsize(16)
        self.btn_duration_up.label.set_color('white')
        self.btn_duration_up.label.set_weight('bold')
        self.btn_duration_up.on_clicked(lambda x: self.change_recording_duration(1))

    def change_recording_duration(self, delta):
        """Change recording duration"""
        new_duration = self.recording_duration + delta
        self.recording_duration = max(1.0, min(10.0, new_duration))
        self.duration_text.set_text(f'{self.recording_duration:.1f}s')

    def update_visualization(self, frame):
        """Update both visualizations and similarity score"""
        try:
            # Update real-time side
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Update file side - Process multiple chunks if needed to maintain speed
            if self.audio_data is not None:
                target_chunks_per_frame = max(1, int(self.playback_speed))
                
                for _ in range(target_chunks_per_frame):
                    chunk, chunk_index = self.get_next_file_chunk()
                    if chunk is not None and chunk_index >= 0:
                        try:
                            # Process file chunk
                            nap_output = self.carfac_file.process_chunk(chunk)
                            sai_output = self.SAI_file.RunSegment(nap_output)
                            self.vis_file.get_vowel_embedding(nap_output)
                            vis_output = self.vis_file.run_frame(sai_output)

                            # Store file SAI frame for comparison
                            self.similarity_calculator.add_file_frame(sai_output)

                            self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                            self.vis_file.draw_column(self.vis_file.img[:, -1])

                            current_file_position = self.current_position
                            if self.current_position == 0 and self.chunks_processed > 0:
                                current_time = self.duration
                            else:
                                current_time = (current_file_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                            
                            if _ == 0:
                                self.process_file_chunk_for_whisper(chunk, current_time)
                            
                            self.chunks_processed += 1
                            
                        except Exception as e:
                            print(f"Error processing file chunk {chunk_index}: {e}")
                            break
                    else:
                        break

            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
            
            # Update text displays
            transcription_rt = self.whisper_realtime.get_display_text()
            if not transcription_rt:
                transcription_rt = "Listening... (speak into microphone)"
            self.transcription_realtime.set_text(transcription_rt)
            
            transcription_file = self.whisper_file.get_display_text()
            self.transcription_file.set_text(transcription_file)
            
            # Update progress
            if self.audio_data is not None:
                progress_pct = (self.current_position / self.total_samples) * 100 if self.total_samples > 0 else 0
                current_time = (self.current_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                loop_info = f" (Loop #{self.loop_count})" if self.loop_count > 0 and self.loop_audio else ""
                progress_text = f"Progress: {progress_pct:.1f}% | {current_time:.1f}s/{self.duration:.1f}s{loop_info}"
                self.progress_file.set_text(progress_text)
            
            # Update status
            if self.is_recording:
                elapsed = time.time() - self.recording_start_time
                remaining = max(0, self.recording_duration - elapsed)
                status_text = f"RECORDING: {remaining:.1f}s remaining"
                self.status_realtime.set_text(status_text)
                self.status_realtime.set_color('red')
            else:
                self.status_realtime.set_text("Ready to record")
                self.status_realtime.set_color('white')
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.im_file, self.transcription_realtime, 
                self.transcription_file, self.progress_file, self.status_realtime,
                self.score_display]

    def start(self):
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
        
        if self.stream:
            self.stream.start_stream()
        
        real_time_interval = (self.chunk_size / self.sample_rate) * 1000
        animation_interval = max(10, int(real_time_interval / max(1, self.playback_speed)))
                
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        plt.show()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.is_recording = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
        try:
            if self.p:
                self.p.terminate()
        except:
            pass

    def stop(self):
        """Stop the system"""
        self.cleanup()
        plt.close('all')

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Dual SAI Pronunciation Trainer with Recording')
    parser.add_argument('--audio-file', default='reference/thai_thankyou_new.mp3', 
                    help='Path to reference audio file')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size (default: 512)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width (default: 400)')
    parser.add_argument('--whisper-model', default='tiny', help='Whisper model (default: tiny)')
    parser.add_argument('--whisper-interval', type=float, default=2.0, help='Whisper processing interval in seconds (default: 2.0)')
    parser.add_argument('--similarity-method', default='cosine', choices=['cosine', 'correlation', 'euclidean', 'spectral'], 
                        help='Similarity calculation method (default: cosine)')
    parser.add_argument('--recording-duration', type=float, default=5.0, help='Default recording duration in seconds (default: 5.0)')
    parser.add_argument('--speed', type=float, default=3.0, help='File playback speed multiplier (default: 3.0)')
    parser.add_argument('--no-loop', action='store_true', help='Disable audio file looping (default: looping enabled)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving recorded audio files (default: save enabled)')
    parser.add_argument('--recording-dir', default='recordings', help='Directory to save recordings (default: recordings)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--language', default='en', choices=['zh', 'vi', 'th', 'en'])
    
    args = parser.parse_args()
    
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        return 1
    
    if not args.audio_file:
        print("Error: No audio file specified")
        return 1
    
    try:
        processor = DualSAIWithRecording(
            audio_file_path=args.audio_file,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            whisper_model=args.whisper_model,
            whisper_interval=args.whisper_interval,
            debug=args.debug,
            playback_speed=args.speed,
            loop_audio=not args.no_loop,
            similarity_method=args.similarity_method,
            save_recordings=not args.no_save,
            recording_dir=args.recording_dir,
            language="vi",
        )
        
        # Set custom recording duration if specified
        processor.set_recording_duration(args.recording_duration)
        
        processor.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        if not args.no_save:
            try:
                summary = processor.get_recordings_summary()
                print(f"\n{summary}")
            except:
                pass
        return 0
    except Exception as e:
        print(f"Error in dual SAI trainer: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) == 1:
        # Get the script's directory (src)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Audio file is in reference subdirectory
        default_audio = os.path.join(script_dir, 'reference', 'vietnamese_thankyou_new.mp3')
        
        if os.path.exists(default_audio):
            sys.argv.append('--audio-file')
            sys.argv.append(default_audio)
        else:
            sys.exit(1)
    sys.exit(main() or 0)